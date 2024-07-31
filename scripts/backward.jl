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

@everywhere begin # Markov chain
    function driftstep(t, idx, F, u::Policy, model::AbstractModel, G)
        L, R = extrema(CartesianIndices(F))
        σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * G.Δ.T))^2
        σₘ² = (model.hogg.σₘ / G.Δ.m)^2

        Xᵢ = G.X[idx]
        dT = μ(Xᵢ.T, Xᵢ.m, model) / (model.hogg.ϵ * G.Δ.T)
        dm = γ(t, model.calibration) - u.α

        # -- Temperature
        FᵢT₊ = F[min(idx + I[1], R)]
        FᵢT₋ = F[max(idx - I[1], L)]
        # -- Carbon concentration
        Fᵢm₊ = F[min(idx + I[2], R)]
        Fᵢm₋ = F[max(idx - I[2], L)]

        Q = σₘ² + σₜ² + G.h * (abs(dT) + abs(dm))

        dTF = G.h * abs(dT) * ifelse(dT > 0, FᵢT₊, FᵢT₋) + σₜ² * (FᵢT₊ + FᵢT₋) / 2
        dmF = G.h * abs(dm) * ifelse(dm > 0, Fᵢm₊, Fᵢm₋) + σₘ² * (Fᵢm₊ + Fᵢm₋) / 2

        F′ = (dTF + dmF) / Q
        Δt = G.h^2 / Q

        return F′, Δt
    end

    markovstep(t, idx, F, u, model::TippingModel, G) = driftstep(t, idx, F, u, model, G)
    function markovstep(t, idx, F, u, model::JumpModel, G)
        Fᵈ, Δt = driftstep(t, idx, F, u, model, G)
    
        # Update with jump
        R = maximum(CartesianIndices(F))
        Xᵢ = G.X[idx]
        πᵢ = intensity(Xᵢ.T, model.hogg, model.jump)
        qᵢ = increase(Xᵢ.T, model.hogg, model.jump)
    
        steps = floor(Int, div(qᵢ, G.Δ.T * G.h))
        weight = qᵢ / (G.Δ.T * G.h)
    
        Fʲ = F[min(idx + steps * I[1], R)] * (1 - weight) + 
                F[min(idx + (steps + 1) * I[1], R)] * weight
    
        F′ = Fᵈ + πᵢ * Δt * (Fʲ - Fᵈ)
    
        return F′, Δt
    end
end

@everywhere begin # Costs
    function cost(F′, t, Xᵢ::Point, Δt, u::Policy, model::AbstractModel{GrowthDamages, P}) where P
        δ = outputfct(t, Xᵢ, Δt, u, model)
        g(u.χ, δ * F′, Δt, model.preferences)
    end

    function cost(F′, t, Xᵢ::Point, Δt, u::Policy, model::AbstractModel{LevelDamages, P}) where P
        δ = outputfct(t, Xᵢ, Δt, u, model)
        damage = d(Xᵢ.T, model.damages, model.hogg)
        g(u.χ * damage, δ * F′, Δt, model.preferences)
    end
end

function backwardstep!(Δts, F, policy, cluster, model, G; allownegative = false)
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

        F[idx] = obj
        policy[idx] = Policy(pol[1], pol[2])

        timestep = last(markovstep(t, idx, F, policy[idx], model, G))
        Δts[i] = timestep
    end
end

"Backward simulates from F̄ down to F₀, using the albedo model. It assumes that the passed F ≡ F̄"
function backwardsimulation!(F, policy, model, G; verbose = false, cachepath = nothing, cachestep = 0.25, overwrite = false, tstop = 0., tcache = last(model.calibration.tspan), allownegative = false)
    verbose && println("Starting backward simulation...")
     
    savecache = !isnothing(cachepath)
    if savecache
        if isfile(cachepath) 
            if overwrite 
                verbose && @warn "Removing file $cachepath.\n"

                rm(cachepath)
            else 
                verbose && @warn "File $cachepath already exists. If you want to overwrite it pass overwrite = true. Will copy the results into `F` and `policy`.\n"

                Fcache, policycache = loadtotal(model, G; allownegative)

                F .= Fcache
                policy .= policycache

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
            backwardstep!(Δts, F, policy, cluster, model, G; allownegative)

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
function computebackward(F̄, terminalpolicy, model, G; verbose = false, withsave = true, datapath = "data", allownegative = false, iterkwargs...) 
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