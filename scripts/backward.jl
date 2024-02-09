using Distributed: @everywhere, @distributed, @sync, workers
using SharedArrays: SharedArray

using Model, Grid

include("../scripts/utils/saving.jl")

@everywhere begin
    using Model, Grid
    using JLD2, DotEnv
    using UnPack: @unpack
    using ZigZagBoomerang: dequeue!
    using Base: Order
    using FastClosures: @closure
    using NLopt: Opt, lower_bounds!, upper_bounds!, min_objective!, optimize
        
    const env = DotEnv.config()
    const DATAPATH = get(env, "DATAPATH", "data")
end


"Backward simulates from V̄ = V(τ) down to V(0). Stores nothing."
function backwardsimulation!(V::SharedArray{Float64, 3}, policy::SharedArray{Policy, 3}, model::ModelInstance, G::RegularGrid; verbose = false, cachepath = nothing, t₀ = 0., cachestep = 0.25)
    verbose && println("Starting backward simulation...")
     
    cache = !isnothing(cachepath)
    if cache
        tcache = model.economy.t₁ # Only cache in the range of the IPCC data forecast
        cachefile = jldopen(cachepath, "a+")
    end

    σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * G.Δ.T))^2
    σₖ² = (model.economy.σₖ / G.Δ.y)^2

    indices = CartesianIndices(G)
    L, R = extrema(indices)

    queue = DiagonalRedBlackQueue(G)
    Δt = SharedArray(zeros(length(queue.vals)))

    while !all(isempty.(queue.minima))
        tmin = model.economy.τ - minimum(queue.vals)
        verbose && print("Cluster minimum time = $tmin\r...")

        cluster = first(dequeue!(queue))

        @sync @distributed for (i, δt) in cluster
            # Constants
            tᵢ = model.economy.τ - δt
            idx = indices[i]
            γₜ = γ(tᵢ, model.economy, model.calibration)
            Xᵢ = G.X[idx]
            Vᵢ = V[idx]

            dT = μ(Xᵢ.T, Xᵢ.m, model.hogg, model.albedo) / model.hogg.ϵ
            δₘᵢ = δₘ(exp(Xᵢ.m), model.hogg)

            supdT = abs(dT)
            supdm = max(γₜ, δₘᵢ)
            supdy = boundb(tᵢ, Xᵢ, model)
            
            supdrift = (supdT / G.Δ.T) + (supdm / G.Δ.m) + (supdy / G.Δ.y)

            Qᵢ = σₜ² + σₖ² + G.h * supdrift
            Δtᵢ = G.h^2 / Qᵢ

            # Neighbouring nodes
            # -- Temperature
            VᵢT₊ = V[min(idx + I[1], R)]
            VᵢT₋ = V[max(idx - I[1], L)]
            # -- Carbon concentration
            Vᵢm₊ = V[min(idx + I[2], R)]
            Vᵢm₋ = V[max(idx - I[2], L)]
            # -- GDP
            Vᵢy₊ = V[min(idx + I[3], R)]
            Vᵢy₋ = V[max(idx - I[3], L)]

            ∂²T = σₜ² * (VᵢT₊ + VᵢT₋) / 2
            ∂²y = σₖ² * (Vᵢy₊ + Vᵢy₋) / 2

            dVT = (G.h / G.Δ.T) * abs(dT) * ifelse(dT > 0, VᵢT₊, VᵢT₋)
            
            negvalue = @closure (x, _) -> begin
                u = Policy(x[1], x[2])

                dy = b(tᵢ, Xᵢ, u, model)
                dVy = (G.h / G.Δ.y) * abs(dy) * ifelse(dy > 0, Vᵢy₊, Vᵢy₋)

                dm = γₜ - u.α
                dVm = (G.h / G.Δ.m) * abs(dm) * ifelse(dm > 0, Vᵢm₊, Vᵢm₋)

                dVᵢ = G.h * ((supdy - abs(dy)) / G.Δ.y + (supdm - abs(dm)) / G.Δ.m) * Vᵢ
            
                EV = (∂²T + dVT + ∂²y + dVy + dVm + dVᵢ) / Qᵢ

                c = u.χ * exp(Xᵢ.y)

                -f(c, EV, Δtᵢ, model.preferences)
            end

            optimiser = Opt(:LN_SBPLX, 2)
            lower_bounds!(optimiser, [0., 0.])
            upper_bounds!(optimiser, [1., γₜ + δₘᵢ])

            min_objective!(optimiser, negvalue)

            candidate = policy[idx]
            alternative = [candidate.χ, ifelse(2candidate.α < γₜ + δₘᵢ, 0.9 * (γₜ + δₘᵢ), 0.1 * γₜ + δₘᵢ)]
            
            obj, pol, _ = optimize(optimiser, candidate)
            objalt, polalt, _ = optimize(optimiser, alternative)
            
            V[idx] = max(-obj, -objalt)
            policy[idx] = ifelse(
                obj < objalt, 
                Policy(pol[1], pol[2]), 
                Policy(polalt[1], polalt[2])
            )

            Δt[i] = ifelse(tᵢ > t₀, G.h^2 / Qᵢ, 0.)
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

function computevalue(N::Int, Δλ, p::Preferences; cache = false, kwargs...)
    name = filename(N, Δλ, p)
    termpath = joinpath(DATAPATH, "terminal", name)
    savepath = joinpath(DATAPATH, "total", name)
    cachepath = cache ? savepath : nothing

    if !isfile(termpath)
        throw("$termpath simulation not found!")
    end
    
    termsim = load(termpath);
    V = SharedArray(termsim["V̄"]);
    terminalpolicy = termsim["policy"];
    model = termsim["model"];
    grid = termsim["G"];
    
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