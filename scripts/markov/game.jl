using Distributed: @everywhere, @distributed, @sync, nprocs
using SharedArrays: SharedArray, SharedMatrix, SharedVector
using DataStructures: PriorityQueue, dequeue!, enqueue!, peek

using Model, Grid

@everywhere begin
    using Model, Grid
    using JLD2, DotEnv
    using UnPack: @unpack
    using ZigZagBoomerang: dequeue!
    using Base: Order
    using FastClosures: @closure
    using NLopt: Opt, lower_bounds!, upper_bounds!, min_objective!, optimize, xtol_rel!
        
    env = DotEnv.config()
    DATAPATH = get(env, "DATAPATH", "data")
end

@everywhere include("chain.jl")

function backwardstep!(Δts, F::AbstractArray{Float64, 4}, policies::AbstractArray{Policy, 4}, cluster, model::AbstractGameModel, G)
    indices = CartesianIndices(G)
    models = breakgamemodel(model)

    τ = models[1].economy.τ
    M = size(F, 3)
    unit = range(0, 1; length = M)

    optimiser = Opt(:LN_SBPLX, 2)
    lower_bounds!(optimiser, [0., 0.]); 

    @sync @distributed for (l, δt) in cluster
        idx = indices[l]
        Xᵢ = G.X[idx]

        t = τ - δt
        δₘₜ = δₘ(exp(Xᵢ.m), model.hogg)
        ᾱ = γ(t, model.regionalcalibration) .+ δₘₜ

        Δtₗ = Inf

        for (i, regionalmodel) in enumerate(models)
            j = ifelse(i == 1, 2, 1)

            xtol_rel!(optimiser, ᾱ[i] / M)
            upper_bounds!(optimiser, [1., ᾱ[i]])

            A = ᾱ[j] * unit

            for (k, αⱼ) in enumerate(A)
                pₖ = @view policies[:, :, k, i]
                Fₖ = @view F[:, :, k, i]
    
                objective = @closure (x, _) -> begin
                    u = Policy(x[1], x[2])
                    F′, Δt = markovstep(t, idx, Fₖ, u, αⱼ, regionalmodel, G)
                    cost(F′, t, Xᵢ, Δt, u, regionalmodel)
                end

                min_objective!(optimiser, objective)
                candidate = min.(pₖ[idx], [1., ᾱ[i]])

                obj, pol, _ = optimize(optimiser, candidate)

                Fₖ[idx] = obj
                pₖ[idx] = pol

                timestep = last(markovstep(t, idx, Fₖ, pₖ[idx], αⱼ, regionalmodel, G))

                Δtₗ = min(timestep, Δtₗ)
            end
        end

        Δts[l] = Δtₗ
    end
end

function backwardsimulation!(
    F::AbstractArray{Float64, 4}, 
    policies::AbstractArray{Policy, 4},
    model::AbstractGameModel, G; 
    verbose = false, cachepath = nothing, cachestep = 0.25, overwrite = false, tstop = 0., tcache = last(model.regionalcalibration.calibration.tspan), allownegative = false, stepkwargs...)
    verbose && println("Starting backward simulation...")
     
    savecache = !isnothing(cachepath)
    if savecache
        if isfile(cachepath)
            if overwrite 
                verbose && @warn "Removing file $cachepath.\n"

                rm(cachepath)
            else 
                verbose && @warn "File $cachepath already exists. If you want to overwrite it pass overwrite = true. Will copy the results into `F` and `policy`.\n"

                _, Fcache, policycache, _, _ = loadtotal(model; allownegative)

                F .= Fcache[:, :, 1]
                policy .= policycache[:, :, 1]

                return F, policy
            end
        end

        cachefile = jldopen(cachepath, "w+")
        cachefile["G"] = G
    end

    queue = DiagonalRedBlackQueue(G)
    Δts = SharedVector(zeros(N^2))
    τ = first(model.economy).τ

    while !isempty(queue)
        tmin = τ - minimum(queue.vals)
        clusters = dequeue!(queue)

        for cluster in clusters
            backwardstep!(Δts, F, policies, cluster, model, G; stepkwargs...)

            verbose && print("Cluster minimum time = $tmin...\r")

            for i in first.(cluster)
                if queue[i] ≤ τ - tstop
                    queue[i] += Δts[i]
                end
            end
        end
        
        if savecache && tmin ≤ tcache
            verbose && println("\nSaving cache at $tcache...")
            group = JLD2.Group(cachefile, "$tcache")
            group["F"] = F
            group["policies"] = policies
            tcache = tcache - cachestep 
        end
    end

    if savecache close(cachefile) end

    return F, policies
end

function computebackward(model::AbstractGameModel, G; datapath = "data", kwargs...)

    h, l = breakgamemodel(model)
    F̄low, pollow, Gterminal = loadterminal(l; addpath = "low", datapath)
    F̄high, polhigh, _ = loadterminal(h; addpath = "high", datapath)

    F̄ = cat(F̄high, F̄low; dims = 3)
    terminalpolicy = cat(polhigh, pollow; dims = 3)
    
    terminalres = F̄, terminalpolicy, Gterminal
    
    computebackward(terminalres, model, G; datapath, kwargs...)
end
function computebackward(terminalres::Tuple{Array{Float64, 3}, Array{Float64, 3}, RegularGrid}, model::AbstractGameModel, G; verbose = false, withsave = true, datapath = "data", allownegative = false, M = 10, iterkwargs...)
    F̄, terminalconsumption, Gterminal = terminalres
    
    N₁, N₂ = size(G)
    nmodels = size(F̄, 3)

    F = SharedArray{Float64}(N₁, N₂, M, nmodels)
    policies = SharedArray{Policy}(N₁, N₂, M, nmodels)

    for k in 1:nmodels
        F[:, :, :, k] .= interpolateovergrid(Gterminal, G, F̄[:, :, k])
        terminalpolicy = [Policy(χ, 0.) for χ ∈ terminalconsumption[:, :, k]]
        policies[:, :, :, k] .= interpolateovergrid(Gterminal, G, terminalpolicy)
    end

    if withsave
        folder = SIMPATHS[typeof(model)]
        controltype = ifelse(allownegative, "allownegative", "nonnegative")
        cachefolder = joinpath(datapath, folder, controltype, "cache")
        if !isdir(cachefolder) mkpath(cachefolder) end
        
        filename = makefilename(model)
    end

    cachepath = withsave ? joinpath(cachefolder, filename) : nothing
    
    backwardsimulation!(F, policies, model, G; verbose, cachepath, allownegative, iterkwargs...)

    return F, policies
end