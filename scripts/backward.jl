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
    using NLopt: Opt, lower_bounds!, upper_bounds!, min_objective!, optimize, xtol_rel!
        
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

        clusters = dequeue!(queue)

        @inbounds for cluster in clusters
            @sync @distributed for (i, δt) ∈ cluster
                idx = indices[i]

                tᵢ = model.economy.τ - δt
                γₜ = γ(tᵢ, model.economy, model.calibration)
                Xᵢ = G.X[idx]

                dT = μ(Xᵢ.T, Xᵢ.m, model.hogg, model.albedo) / (model.hogg.ϵ * G.Δ.T)
                δₜ = δₘ(exp(Xᵢ.m), model.hogg)

                supdT = abs(dT)
                supdm = max(γₜ, δₜ) / G.Δ.m
                supdy = boundb(tᵢ, Xᵢ, model) / G.Δ.y
                
                # Time speed
                Qᵢ = σₜ² + σₖ² + G.h * (supdT + supdm + supdy)
                Δtᵢ = G.h^2 / Qᵢ

                # Neighbouring nodes
                Vᵢ = V[idx]
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

                dVT = G.h * abs(dT) * ifelse(dT > 0, VᵢT₊, VᵢT₋)
                
                negvalue = @closure (x, _) -> begin
                    u = Policy(x[1], x[2])

                    dy = b(tᵢ, Xᵢ, u, model) / G.Δ.y
                    dVy = G.h * abs(dy) * ifelse(dy > 0, Vᵢy₊, Vᵢy₋)

                    dm = (γₜ - u.α) / G.Δ.m
                    dVm = G.h * abs(dm) * ifelse(dm > 0, Vᵢm₊, Vᵢm₋)

                    dVᵢ = G.h * (supdy - abs(dy) + supdm - abs(dm)) * Vᵢ
                
                    v = (∂²T + dVT + ∂²y + dVy + dVm + dVᵢ) / Qᵢ

                    c = u.χ * exp(Xᵢ.y)

                    -f(c, v, Δtᵢ, model.preferences)
                end

                ᾱ = γₜ + δₜ

                optimiser = Opt(:LN_SBPLX, 2)
                lower_bounds!(optimiser, [0., 0.])
                upper_bounds!(optimiser, [1., ᾱ])
                xtol_rel!(optimiser, 1e-3)

                min_objective!(optimiser, negvalue)

                candidate = clamp.(policy[idx], [0., 0.], [1., ᾱ])
                alternative = [candidate[1], ifelse(candidate[2] < (ᾱ / 2), 0.9 * ᾱ, 0.1 * ᾱ)]
                
                obj, pol, _ = optimize(optimiser, candidate)            
                objalt, polalt, _ = optimize(optimiser, alternative)
                
                V[idx] = max(-obj, -objalt)
                policy[idx] = ifelse(-obj > -objalt, Policy(pol[1], pol[2]), Policy(polalt[1], polalt[2]))

                Δt[i] = ifelse(tᵢ > t₀, Δtᵢ, 0.)
            end
        end

        for cluster in clusters, (i, _) in cluster
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
    name = makefilename(N, Δλ, p)
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
    G = termsim["G"];
    
    policy = SharedArray([Policy(χ, 0.) for χ ∈ terminalpolicy]);

    if cache
        println("Saving in $cachepath...")
    end
    
    backwardsimulation!(V, policy, model, G; cachepath = cachepath, kwargs...)
    
    println("\nSaving solution into $savepath...")
    jldopen(savepath, "a+") do cachefile 
        g = JLD2.Group(cachefile, "endpoint")
        g["V"] = V
        g["policy"] = policy
        g["G"] = G
    end
    
    return V, policy
end