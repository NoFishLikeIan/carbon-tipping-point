using Distributed: @everywhere, @distributed, @sync, workers
using SharedArrays: SharedArray, SharedMatrix
using DataStructures: PriorityQueue, dequeue!, enqueue!, peek

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

"Backward simulates from F̄ down to F₀, using the albedo model. It assumes that the passed F ≡ F̄"
function backwardsimulation!(F::SharedMatrix{Float64}, policy::SharedMatrix{Policy}, model::ModelInstance, G::RegularGrid; verbose = false, cachepath = nothing, t₀ = 0., cachestep = 0.25)
    verbose && println("Starting backward simulation...")
     
    docache = !isnothing(cachepath)
    if docache
        tcache = model.economy.t₁ # Caches only the IPCC forecast timespan
        isfile(cachepath) && throw("File $cachepath already exists.")
        cachefile = jldopen(cachepath, "w+")
    end

    σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * G.Δ.T))^2
    σₘ² = (model.hogg.σₘ / G.Δ.m)^2
    σₖ² = model.economy.σₖ^2

    indices = CartesianIndices(G)
    L, R = extrema(indices)

    queue = DiagonalRedBlackQueue(G)
    Δt = SharedArray(zeros(length(queue.vals)))

    while !all(isempty.(queue.minima))
        tmin = model.economy.τ - minimum(queue.vals)
        verbose && print("Cluster minimum time = $tmin\r...")

        clusters = dequeue!(queue)

        @inbounds for cluster in clusters
            @sync @distributed for (i, δt) in cluster
                idx = indices[i]

                tᵢ = model.economy.τ - δt
                # FIXME: γₜ = 0 for t >> 0
                γₜ = γ(tᵢ, model.economy, model.calibration)
                Xᵢ = G.X[idx]

                dT = μ(Xᵢ.T, Xᵢ.m, model.hogg, model.albedo) / (model.hogg.ϵ * G.Δ.T)
                δₜ = δₘ(exp(Xᵢ.m), model.hogg)

                supdT = abs(dT)
                supdm = max(γₜ, δₜ) / G.Δ.m
                
                # Time speed
                Qᵢ = σₜ² + σₘ² + G.h * (supdT + supdm)
                Δtᵢ = G.h^2 / Qᵢ

                # Neighbouring nodes
                Fᵢ = F[idx]
                # -- Temperature
                FᵢT₊ = F[min(idx + I[1], R)]
                FᵢT₋ = F[max(idx - I[1], L)]
                # -- Carbon concentration
                Fᵢm₊ = F[min(idx + I[2], R)]
                Fᵢm₋ = F[max(idx - I[2], L)]

                ∂²T = σₜ² * (FᵢT₊ + FᵢT₋) / 2
                ∂²m = σₘ² * (Fᵢm₊ + Fᵢm₋) / 2

                dFT = G.h * abs(dT) * ifelse(dT > 0, FᵢT₊, FᵢT₋)
                
                costs = @closure (x, _) -> begin
                    u = Policy(x[1], x[2])

                    μy = b(tᵢ, Xᵢ, u, model)

                    dm = (γₜ - u.α) / G.Δ.m
                    dFm = G.h * abs(dm) * ifelse(dm > 0, Fᵢm₊, Fᵢm₋)

                    dFᵢ = G.h * (supdm - abs(dm)) * Fᵢ # ∝ prob of staying
                
                    F′ = (∂²T + ∂²m + dFT + dFm + dFᵢ) / Qᵢ

                    δy = max(1 + (1 - model.preferences.θ) * (μy - model.preferences.θ * σₖ²) * Δtᵢ, 0.)

                    return g(u.χ, δy * F′, Δtᵢ, model.preferences)
                end

                ᾱ = γₜ + δₜ

                optimiser = Opt(:LN_SBPLX, 2)
                lower_bounds!(optimiser, [0., 0.])
                upper_bounds!(optimiser, [1., ᾱ])
                xtol_rel!(optimiser, 1e-3)

                min_objective!(optimiser, costs)

                candidate = clamp.(policy[idx], [0., 0.], [1., ᾱ])
                obj, pol, _ = optimize(optimiser, candidate)
                
                F[idx] = obj
                policy[idx] = Policy(pol[1], pol[2])

                Δt[i] = ifelse(tᵢ > t₀, Δtᵢ, 0.)
            end
        end

        for cluster in clusters, i in first.(cluster)
            if Δt[i] > 0.
                queue[i] += Δt[i]
            end
        end
        
        if docache && tmin ≤ tcache
            verbose && println("\nSaving cache at $tcache...")
            group = JLD2.Group(cachefile, "$tcache")
            group["F"] = F
            group["policy"] = policy
            tcache = tcache - cachestep 
        end
    end

    if docache close(cachefile) end

    return F, policy
end

# FIXME: The red-black parallelisation does not work because V at position x depends on the jump size q(x). This is not an issue as long as |q(x)| is sufficiently small. I do not check this.
"Backward simulates from V̄ down to V₀, using the jump model."
function backwardsimulation!(F::SharedMatrix{Float64}, policy::SharedMatrix{Policy}, model::ModelBenchmark, G::RegularGrid; verbose = false, cachepath = nothing, t₀ = 0., cachestep = 0.25)
    verbose && println("Starting backward simulation...")
    
    docache = !isnothing(cachepath)
    if docache
        tcache = model.economy.t₁ # Caches only the IPCC forecast timespan
        isfile(cachepath) && throw("File $cachepath already exists.")
        cachefile = jldopen(cachepath, "w+")
    end

    σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * G.Δ.T))^2
    σₘ² = (model.hogg.σₘ / G.Δ.m)^2
    σₖ² = model.economy.σₖ^2

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

                dT = μ(Xᵢ.T, Xᵢ.m, model.hogg) / (model.hogg.ϵ * G.Δ.T)
                δₜ = δₘ(exp(Xᵢ.m), model.hogg)

                supdT = abs(dT)
                supdm = max(γₜ, δₜ) / G.Δ.m
                
                # Time speed
                Qᵢ = σₜ² + σₘ² + G.h * (supdT + supdm)
                Δtᵢ = G.h^2 / Qᵢ

                # Jump
                πᵢ = intensity(Xᵢ.T, model.hogg, model.jump)
                qᵢ = increase(Xᵢ.T, model.hogg, model.jump)

                steps = floor(Int, div(qᵢ, G.Δ.T * G.h))
                weight = qᵢ / (G.Δ.T * G.h)

                Fʲ = F[min(idx + steps * I[1], R)] * (1 - weight) + F[min(idx + (steps + 1) * I[1], R)] * weight

                # Neighbouring nodes
                Fᵢ = F[idx]
                # -- Temperature
                FᵢT₊ = F[min(idx + I[1], R)]
                FᵢT₋ = F[max(idx - I[1], L)]
                # -- Carbon concentration
                Fᵢm₊ = F[min(idx + I[2], R)]
                Fᵢm₋ = F[max(idx - I[2], L)]

                ∂²T = σₜ² * (FᵢT₊ + FᵢT₋) / 2
                ∂²m = σₘ² * (Fᵢm₊ + Fᵢm₋) / 2

                dFT = G.h * abs(dT) * ifelse(dT > 0, FᵢT₊, FᵢT₋)
                
                costs = @closure (x, _) -> begin
                    u = Policy(x[1], x[2])
                    μy = b(tᵢ, Xᵢ, u, model)

                    dm = (γₜ - u.α) / G.Δ.m
                    dFm = G.h * abs(dm) * ifelse(dm > 0, Fᵢm₊, Fᵢm₋)

                    dFᵢ = G.h * (supdm - abs(dm)) * Fᵢ # ∝ prob of staying
                
                    Fᵈ = (∂²T + ∂²m + dFT + dFm + dFᵢ) / Qᵢ
                    
                    F′ = Fᵈ + πᵢ * Δtᵢ * (Fʲ - Fᵈ)

                    δy = max(1 + (1 - model.preferences.θ) * (μy - model.preferences.θ * σₖ²) * Δtᵢ, 0.)

                    return g(u.χ, δy * F′, Δtᵢ, model.preferences)
                end

                ᾱ = γₜ + δₜ

                optimiser = Opt(:LN_SBPLX, 2)
                lower_bounds!(optimiser, [0., 0.])
                upper_bounds!(optimiser, [1., ᾱ])
                xtol_rel!(optimiser, 1e-3)

                min_objective!(optimiser, costs)

                candidate = clamp.(policy[idx], [0., 0.], [1., ᾱ])
                obj, pol, _ = optimize(optimiser, candidate)
                
                F[idx] = obj
                policy[idx] = Policy(pol[1], pol[2])

                Δt[i] = ifelse(tᵢ > t₀, Δtᵢ, 0.)
            end
        end

    for cluster in clusters, i in first.(cluster)
            if Δt[i] > 0.
                queue[i] += Δt[i]
            end
        end
        
        if docache && tmin ≤ tcache
            verbose && println("\nSaving cache at $tcache...")
            group = JLD2.Group(cachefile, "$tcache")
            group["F"] = F
            group["policy"] = policy
            tcache = tcache - cachestep 
        end
    end

    if docache close(cachefile) end

    return F, policy
end

function computevalue(model, G::RegularGrid; docache = false, verbose  = false, kwargs...)
    folder = typeof(model) <: ModelInstance ? "albedo" : "jump"
    filename = makefilename(model, G)
    savepath = joinpath(DATAPATH, folder, "backward", filename)
    cachepath = docache ? savepath : nothing
    
    F̄, terminalpolicy = loadterminal(model, G; datapath = DATAPATH)
    policy = SharedMatrix([Policy(χ, 0.) for χ ∈ terminalpolicy]);

    F = SharedMatrix(F̄)

    docache && verbose  && println("Saving in $cachepath...")
    
    backwardsimulation!(F, policy, model, G; cachepath = cachepath, verbose, kwargs...)

    verbose && println("Done!")
    return F, policy
end