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
end

@everywhere include("chain.jl")

function backwardstep!(Δts, F, policy, cluster, model::AbstractModel, G; allownegative = false, s = 1e-2)
    indices = CartesianIndices(G)

    @sync @distributed for (i, δt) in cluster
        idx = indices[i]
        Xᵢ = G.X[idx]

        t = model.economy.τ - δt
        ᾱ = allownegative ? 1. : 
            γ(t, model.calibration) + δₘ(exp(Xᵢ.m), model.hogg)

        objective = @closure (x, grad) -> begin
            u = Policy(x[1], x[2]) 
            F′, Δt = markovstep(t, idx, F, u, model, G)
            cost(F′, t, Xᵢ, Δt, u, model)
        end

        optimiser = Opt(:LN_SBPLX, 2); xtol_rel!(optimiser, ᾱ / 100.);
	    lower_bounds!(optimiser, [0., 0.]); upper_bounds!(optimiser, [1., ᾱ])
        min_objective!(optimiser, objective)

        candidate = min.(policy[idx], [1., ᾱ])
        obj, pol, _ = optimize(optimiser, candidate)
        
        polₜ = Policy(pol[1], pol[2])
        timestep = last(markovstep(t, idx, F, polₜ, model, G))
        
        w = inv(1 + s * timestep) # Policy smoothing over time

        policy[idx] = adjpolicy(idx, policy) * (1 - w) + polₜ * w
        F[idx] = obj
        Δts[i] = timestep
    end
end

"Backward simulates from F̄ down to F₀, using the albedo model. It assumes that the passed F ≡ F̄"
function backwardsimulation!(F, policy, model::AbstractModel, G; verbose = false, cachepath = nothing, cachestep = 0.25, overwrite = false, tstop = 0., tcache = last(model.calibration.tspan), allownegative = false, stepkwargs...)
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
    Δts = SharedVector(zeros(length(queue.vals)))

    while !all(isempty.(queue.minima))
        tmin = model.economy.τ - minimum(queue.vals)
        verbose && print("Cluster minimum time = $tmin...\r")

        clusters = dequeue!(queue)

        for cluster in clusters
            backwardstep!(Δts, F, policy, cluster, model, G; allownegative, stepkwargs...)

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

function computebackward(model::AbstractModel, G; datapath = "data", kwargs...)
    F̄, terminalpolicy = loadterminal(model, G)
    computebackward(F̄, terminalpolicy, model, G; datapath, kwargs...)
end
function computebackward(F̄, terminalpolicy, model::AbstractModel, G; verbose = false, withsave = true, datapath = "data", allownegative = false, iterkwargs...) 
    F = SharedMatrix(F̄);
    policy = SharedMatrix([Policy(χ, 0.) for χ ∈ terminalpolicy])

    if withsave
        folder = SIMPATHS[typeof(model)]
        controltype = ifelse(allownegative, "allownegative", "nonnegative")
        cachefolder = joinpath(datapath, folder, controltype, "cache")
        if !isdir(cachefolder) mkpath(cachefolder) end
        
        filename = makefilename(model, G)
    end

    cachepath = ifelse(withsave, joinpath(cachefolder, filename), nothing)
    backwardsimulation!(F, policy, model, G; verbose, cachepath, allownegative, iterkwargs...)

    return F, policy
end