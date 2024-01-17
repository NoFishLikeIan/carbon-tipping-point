using Distributed: @everywhere, @distributed, @sync
using SharedArrays: SharedArray

@everywhere begin
    using Model, Grid
    using JLD2, DotEnv
    using UnPack: @unpack
    using ZigZagBoomerang: dequeue!
    using Base: Order
    using FastClosures: @closure
    using BlackBoxOptim: bboptimize, best_candidate, best_fitness
    
    const env = DotEnv.config()
    const DATAPATH = get(env, "DATAPATH", "data")
end


"Backward simulates from V̄ = V(τ) down to V(0). Stores nothing."
function backwardsimulation!(V::SharedArray{Float64, 3}, policy::SharedArray{Policy, 3}, model::ModelInstance, grid::RegularGrid; verbose = false, cachepath = nothing, tmin = 0., Δtcache = 0.25)    
    verbose && println("Starting backward simulation...")
    cache = !isnothing(cachepath)
    if cache
        tcache = model.economy.τ
        cachefile = jldopen(cachepath, "a+")

        g = JLD2.Group(cachefile, "$(floor(Int, tcache))")
        g["V"] = V
        g["policy"] = policy
        tcache = tcache - Δtcache
    end

    σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * grid.Δ.T))^2
    σₖ² = (model.economy.σₖ / grid.Δ.y)^2

    indices = CartesianIndices(grid)
    L, R = extrema(indices)

    queue = DiagonalRedBlackQueue(grid)

    while !all(isempty.(queue.minima))
        tcluster = model.economy.τ - minimum(queue.vals)
        verbose && println("\nCluster minimum time = $tcluster...")

        cluster = first(dequeue!(queue))

        @sync @distributed for (i, δt) in cluster
            tᵢ = model.economy.τ - δt 
            idx = indices[i]

            Xᵢ = grid.X[idx]
            dT = μ(Xᵢ.T, Xᵢ.m, model.hogg, model.albedo) / (model.hogg.ϵ * grid.Δ.T)

            # Neighbouring nodes
            # -- Temperature
            VᵢT₊ = V[min(idx + I[1], R)]
            VᵢT₋ = V[max(idx - I[1], L)]
            # -- Carbon concentration
            Vᵢm₊ = V[min(idx + I[2], R)]
            # -- GDP
            Vᵢy₊ = V[min(idx + I[3], R)]
            Vᵢy₋ = V[max(idx - I[3], L)]

            γₜ = γ(tᵢ, model.economy, model.calibration)

            negvalue = @closure u -> begin
                χ, α = u
                dy = b(tᵢ, Xᵢ, χ, α, model) / grid.Δ.y
                dm = (γₜ - α) / grid.Δ.m

                Q = σₜ² + σₖ² + grid.h * (abs(dT) + abs(dy) + dm)

                py₊ = (grid.h * max(dy, 0.) + σₖ² / 2) / Q
                py₋ = (grid.h * max(-dy, 0.) + σₖ² / 2) / Q

                pT₊ = (grid.h * max(dT, 0.) + σₜ² / 2) / Q
                pT₋ = (grid.h * max(-dT, 0.) + σₜ² / 2) / Q

                pm₊ = grid.h * dm / Q

                EV̄ = py₊ * Vᵢy₊ + py₋ * Vᵢy₋ + pT₊ * VᵢT₊ + pT₋ * VᵢT₋ + pm₊ * Vᵢm₊

                Δt = grid.h^2 / Q

                return -f(χ, Xᵢ, EV̄, Δt, model.economy)
            end
            
            bounds = [(0., 1.), (0., γₜ)]

            leftres = bboptimize(negvalue, [0., 0.]; SearchRange = bounds, TraceMode = :silent)
            rightres = bboptimize(negvalue, [1., γₜ]; SearchRange = bounds, TraceMode = :silent)

            V[idx] = -min(best_fitness(leftres), best_fitness(rightres))
            policy[idx] = ifelse(best_fitness(leftres) < best_fitness(rightres),best_candidate(leftres), best_candidate(rightres))

            if tᵢ > tmin
                χ, α = policy[idx]
                dy = b(tᵢ, Xᵢ, χ, α, model) / grid.Δ.y
                dm = (γₜ - α) / grid.Δ.m
    
                Q = σₜ² + σₖ² + grid.h * (abs(dT) + abs(dy) + dm)
                Δt = model.grid.h^2 / Q
    
                queue[i] += Δt
            end
        end
        
        if cache && tcluster ≤ tcache 
            verbose && println("\nSaving cache at $(floor(Int, tcache))...")
            g = JLD2.Group(cachefile, "$(floor(Int, tcache))")
            g["V"] = V
            g["policy"] = policy
            tcache = tcache - Δtcache 
        end
    end

    if cache
        close(cachefile)
    end

    return V, policy
end

function computevalue(N::Int, Δλ = 0.08; cache = false, kwargs...)
    filename = "N=$(N)_Δλ=$(Δλ).jld2"
    termpath = joinpath(DATAPATH, "terminal", filename)

    if !isfile(termpath)
        throw("$termpath simulation not found!")
    end

    savepath = joinpath(DATAPATH, "total", filename)
    cachepath = cache ? savepath : nothing

    termsim = load(termpath)
    V̄ = SharedArray(termsim["V̄"]);
    terminalpolicy = termsim["policy"];
    model = termsim["model"];
    grid = termsim["grid"];
    
    policy = SharedArray([Policy(χ, 0.) for χ ∈ terminalpolicy]);
    V = deepcopy(V̄);

    backwardsimulation!(V, policy, model; cachepath, kwargs...)
    
    println("\nSaving solution into $savepath...")
    jldopen(savepath, "a+") do cachefile 
        g = JLD2.Group(cachefile, "endpoint")
        g["V"] = V
        g["policy"] = policy
    end
    
    return V, policy
end