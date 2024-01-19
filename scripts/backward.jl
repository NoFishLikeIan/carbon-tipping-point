using Distributed: @everywhere, @distributed, @sync, workers
using SharedArrays: SharedArray

@everywhere begin
    using Model, Grid
    using JLD2, DotEnv
    using UnPack: @unpack
    using ZigZagBoomerang: dequeue!
    using Base: Order
    using FastClosures: @closure
    using BlackBoxOptim: bboptimize, best_candidate, best_fitness, adaptive_de_rand_1_bin_radiuslimited
        
    const env = DotEnv.config()
    const DATAPATH = get(env, "DATAPATH", "data")
end


"Backward simulates from V̄ = V(τ) down to V(0). Stores nothing."
function backwardsimulation!(V::SharedArray{Float64, 3}, policy::SharedArray{Policy, 3}, model::ModelInstance, grid::RegularGrid; verbose = false, cachepath = nothing, t₀ = 0., cachestep = 0.25)
    verbose && println("Starting backward simulation...")
     
    cache = !isnothing(cachepath)
    if cache
        tcache = model.economy.τ - cachestep
        cachefile = jldopen(cachepath, "a+")
    end

    σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * grid.Δ.T))^2
    σₖ² = (model.economy.σₖ / grid.Δ.y)^2

    indices = CartesianIndices(grid)
    L, R = extrema(indices)

    queue = DiagonalRedBlackQueue(grid)
    Δt = SharedArray(zeros(length(queue.vals)))

    while !all(isempty.(queue.minima))
        tmin = model.economy.τ - minimum(queue.vals)
        verbose && print("Cluster minimum time = $tmin\r...")

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

                Δtᵢ = grid.h^2 / Q

                return -f(χ, Xᵢ, EV̄, Δtᵢ, model.economy)
            end
            
            bounds = [(0., 1.), (0., γₜ)]

            res = bboptimize(negvalue; SearchRange = bounds, TraceMode = :silent, Method = :adaptive_de_rand_1_bin_radiuslimited)

            V[idx] = -best_fitness(res)
            policy[idx] = best_candidate(res)

            if tᵢ > t₀
                dy = b(tᵢ, Xᵢ, policy[idx], model) / grid.Δ.y
                dm = (γₜ - policy[idx].α) / grid.Δ.m
    
                Q = σₜ² + σₖ² + grid.h * (abs(dT) + abs(dy) + dm)
                Δtᵢ = grid.h^2 / Q
    
                Δt[i] = Δtᵢ
            else 
                Δt[i] = 0.
            end
        end

        for i in first.(cluster)
            if Δt[i] > 0.
                queue[i] += Δt[i]
            end
        end
        
        if cache && tmin ≤ tcache
            verbose && println("\nSaving cache at $tcache...")
            g = JLD2.Group(cachefile, "$tcache")
            g["V"] = V
            g["policy"] = policy
            tcache = tcache - cachestep 
        end
    end

    if cache
        close(cachefile)
    end

    return V, policy
end

function computevalue(N::Int, Δλ; cache = false, kwargs...)
    filename = "N=$(N)_Δλ=$(Δλ).jld2"
    termpath = joinpath(DATAPATH, "terminal", filename)
    savepath = joinpath(DATAPATH, "total", filename)
    cachepath = cache ? savepath : nothing

    if !isfile(termpath)
        throw("$termpath simulation not found!")
    end
    
    termsim = load(termpath);
    V = SharedArray(termsim["V̄"]);
    terminalpolicy = termsim["policy"];
    model = termsim["model"];
    grid = termsim["grid"];
    
    policy = SharedArray([Policy(χ, 0.) for χ ∈ terminalpolicy]);

    if cache
        println("Saving in $cachepath...")
    end
    
    backwardsimulation!(V, policy, model, grid; cachepath = cachepath, kwargs...)
    
    println("\nSaving solution into $savepath...")
    jldopen(savepath, "a+") do cachefile 
        g = JLD2.Group(cachefile, "endpoint")
        g["V"] = V
        g["policy"] = policy
        g["grid"] = grid
    end
    
    return V, policy
end