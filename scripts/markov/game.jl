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

function backwardstep!(Δts, Fs::NTuple{2, AbstractArray{Float64, 3}}, policies::NTuple{2, AbstractArray{Policy, 3}}, cluster, model::AbstractGameModel, G)
    indices = CartesianIndices(G)
    highmodel, lowmodel = breakgamemodel(model)

    @sync @distributed for (i, δt) in cluster
        idx = indices[i]
        Xᵢ = G.X[idx]

        t = first(model.economy).τ - δt
        δₘₜ = δₘ(exp(Xᵢ.m), model.hogg)
        ᾱₕ, ᾱₗ = γ(t, model.regionalcalibration) .+ δₘₜ

        optimiserₕ = Opt(:LN_SBPLX, 2); xtol_rel!(optimiserₕ, ᾱₕ / 100.);
	    lower_bounds!(optimiserₕ, [0., 0.]); upper_bounds!(optimiserₕ, [1., ᾱₕ])
        
        Aₗ = ᾱₗ * range(0, 1; length = size(Fs[1], 3))
        for (k, αₗ) in enumerate(Aₗ)
            pₖ = @view policies[1][:, :, k]
            Fₖ = @view Fs[1][:, :, k]

            objective = @closure (x, grad) -> begin
                u = Policy(x[1], x[2])
                F′, Δt = markovstep(t, idx, Fₖ, u, αₗ, model, G)
                cost(F′, t, Xᵢ, Δt, u, highmodel)
            end

            min_objective!(optimiserₕ, objective)
            candidate = min.(pₖ[idx], [1., ᾱₕ])

            obj, pol, _ = optimize(optimiserₕ, candidate)

            Fₖ[idx] = obj
            pₖ[idx] = pol

            timestep = last(markovstep(t, idx, Fₖ, pₖ[idx], αₗ, model, G))
            Δts[i] = timestep
        end

        optimiserₗ = Opt(:LN_SBPLX, 2); xtol_rel!(optimiserₗ, ᾱₗ / 100.);
	    lower_bounds!(optimiserₗ, [0., 0.]); upper_bounds!(optimiserₗ, [1., ᾱₗ])

        Aₕ = ᾱₕ * range(0, 1; length = size(Fs[2], 3))
        for (k, αₕ) in enumerate(Aₕ)
            pₖ = @view policies[2][:, :, k]
            Fₖ = @view Fs[2][:, :, k]

            objective = @closure (x, grad) -> begin
                u = Policy(x[1], x[2])
                F′, Δt = markovstep(t, idx, Fₖ, u, αₕ, model, G)
                cost(F′, t, Xᵢ, Δt, u, lowmodel)
            end

            min_objective!(optimiserₕ, objective)
            candidate = min.(pₖ[idx], [1., ᾱₕ])

            obj, pol, _ = optimize(optimiserₕ, candidate)

            Fₖ[idx] = obj
            pₖ[idx] = pol

            timestep = last(markovstep(t, idx, Fₖ, pₖ[idx], αₕ, model, G))
            Δts[i] = min(timestep, Δts[i])
        end
    end
end

function backwardsimulation!(
    Fs::NTuple{2, AbstractArray{Float64, 3}}, 
    policies::NTuple{2, AbstractArray{Policy, 3}}, 
    model::AbstractGameModel, 
    G; 
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

                _, Fcache, policycache = loadtotal(model, G; allownegative)

                F .= Fcache[:, :, 1]
                policy .= policycache[:, :, 1]

                return F, policy
            end
        end

        cachefile = jldopen(cachepath, "w+")
    end

    queue = DiagonalRedBlackQueue(G)
    Δts = SharedVector(zeros(N^2))

    while !isqempty(queue)
        tmin = first(model.economy).τ - minimum(queue.vals)

        clusters = dequeue!(queue)

        for cluster in clusters
            backwardstep!(Δts, Fs, policies, cluster, model, G; stepkwargs...)

            for i in first.(cluster)
                if queue[i] ≤ model.economy.τ - tstop
                    queue[i] += Δts[i]
                end
            end
        end
        
        if savecache && tmin ≤ tcache
            verbose && println("\nSaving cache at $tcache...")
            group = JLD2.Group(cachefile, "$tcache")
            group["F"] = F
            group["policy"] = policy
            tcache = tcache - cachestep 
        end
    end

    if savecache close(cachefile) end

    return F, policy
end

function computebackward(model, G; kwargs...)
    F̄, terminalpolicy = loadterminal(model, G)
    computebackward(F̄, terminalpolicy, model, G; kwargs...)
end
function computebackward(F̄::Array{Float64, 3}, terminalpolicy::Array{Float64, 3}, model::AbstractModel, G; verbose = false, withsave = true, datapath = "data", allownegative = false, M = 10, iterkwargs...)
    N₁, N₂ = size(G)
    nmodels = size(F̄, 3)

    Fs = ntuple(_ -> SharedArray{Float64}(N₁, N₂, M), nmodels)
    policies = ntuple(_ -> SharedArray{Policy}(N₁, N₂, M), nmodels)

    for m in eachindex(Fs), k in axes(Fs[m], 3)
        Fs[m][:, :, k] .= F̄[:, :, m]
        policies[m][:, :, k] .= [Policy(χ, 0.) for χ ∈ terminalpolicy[:, :, m]]
    end

    if withsave
        folder = SIMPATHS[typeof(model)]
        controltype = ifelse(allownegative, "allownegative", "nonnegative")
        cachefolder = joinpath(datapath, folder, controltype, "cache")
        if !isdir(cachefolder) mkpath(cachefolder) end
        
        filename = makefilename(model, G)
    end

    cachepath = ifelse(withsave, joinpath(cachefolder, filename), nothing)
    backwardsimulation!(Fs, policies, model, G; verbose, cachepath, allownegative, iterkwargs...)

    return F, policy
end