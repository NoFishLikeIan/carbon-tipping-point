using Model, Grid

using JLD2, DotEnv
using UnPack: @unpack
using Polyester: @batch
using FastClosures: @closure
using Roots: find_zero

include("utils/saving.jl")

"Computes the Jacobi iteration for the terminal problem, V̄."
function terminaljacobi!(V̄::AbstractArray{Float64, 3}, policy::AbstractArray{Float64, 3}, model::ModelInstance, G::RegularGrid; indices = CartesianIndices(G))

    L, R = extrema(CartesianIndices(G))
    
    @batch for idx in indices
        σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * G.Δ.T))^2
        σₖ² = (model.economy.σₖ / G.Δ.y)^2
        Xᵢ = G.X[idx]
        dT = μ(Xᵢ.T, Xᵢ.m, model.hogg, model.albedo) / (model.hogg.ϵ * G.Δ.T)

        # Neighbouring nodes
        # -- GDP
        Vᵢy₊ = V̄[min(idx + I[3], R)]
        Vᵢy₋ = V̄[max(idx - I[3], L)]

        # -- Temperature
        VᵢT₊ = V̄[min(idx + I[1], R)]
        VᵢT₋ = V̄[max(idx - I[1], L)]

        noisey = σₖ² * (Vᵢy₊ + Vᵢy₋) / 2
        noiseT = σₜ² * (VᵢT₊ + VᵢT₋) / 2

        dVT = G.h * abs(dT) * ifelse(dT > 0, VᵢT₊, VᵢT₋) + noiseT
        
        value = @closure χ -> begin
            dy = bterminal(Xᵢ, χ, model) / G.Δ.y
            dVy = G.h * abs(dy) * ifelse(dy > 0, Vᵢy₊, Vᵢy₋) + noisey
            
            # Expected value
            Q = σₜ² + σₖ² + G.h * (abs(dT) + abs(dy))
            EV = (dVy + dVT) / Q
            Δt = G.h^2 / Q

            c = χ * exp(Xᵢ.y)

            f(c, EV, Δt, model.preferences)
        end

        # Optimal control
        v, χ = gss(value, 0., 1.)
        
        policy[idx] = χ
        V̄[idx] = v
    end

    return V̄, policy
end

function vfi(V₀::AbstractArray{Float64, 3}, model::ModelInstance, G::RegularGrid; tol = 1e-3, maxiter = 100_000, verbose = false, indices = CartesianIndices(G), alternate = false)
    pᵢ = 0.5 .* ones(size(G))
    pᵢ₊₁ = copy(pᵢ)

    Vᵢ = copy(V₀)
    Vᵢ₊₁ = copy(V₀)
    
    verbose && println("Starting iterations...")
    for iter in 1:maxiter
        terminaljacobi!(
            Vᵢ₊₁, pᵢ₊₁, model, G; 
            indices = (alternate && isodd(iter)) ? indices : reverse(indices) 
        )

        ε = maximum(abs.((Vᵢ₊₁ .- Vᵢ) ./ Vᵢ))
        α = maximum(abs.(pᵢ₊₁ .- pᵢ) ./ pᵢ)

        if verbose
            print("Iteration $iter / $maxiter, ε = $ε and α = $α...\r")
        end

        if ε < tol
            verbose && println("\nDone with convergence, ε = $ε and α = $α.")
            return Vᵢ₊₁, pᵢ₊₁
        end
        
        Vᵢ .= Vᵢ₊₁
        pᵢ .= pᵢ₊₁
    end

    verbose && println("\nDone without convergence, , ε = $ε and α = $α..")
    return Vᵢ₊₁, pᵢ₊₁
end

function computeterminal(N::Int, Δλ, preferences::Preferences; verbose = true, withsave = true, datapath = "data", normalise = false, iterkwargs...)
    economy = Economy()
    hogg = Hogg()
    albedo = Albedo(λ₂ = Albedo().λ₁ - Δλ)
    calibration = load_object(joinpath(datapath, "calibration.jld2"))

    model = ModelInstance(preferences, economy, hogg, albedo, calibration)

    domains = [
        (hogg.T₀, hogg.T₀ + 4.), 
        (mstable(hogg.T₀, hogg, albedo), mstable(hogg.T₀ + 4., hogg, albedo)),
        (log(economy.Y₀ / 2), log(2economy.Y₀)), 
    ]
    
    G = RegularGrid(domains, N);
    
    Vcurve = [log(exp(Xᵢ.y)) / preferences.ρ for Xᵢ ∈ G.X]

    V₀ = typeof(preferences) <: EpsteinZin ? 
        Vcurve .- 2maximum(Vcurve) : Vcurve

    V̄, policy = vfi(V₀, model, G; verbose, iterkwargs...)

    V̄ = normalise ? V̄ ./ minimum(abs.(V̄)) : V̄

    if withsave
        savepath = joinpath(datapath, "terminal", filename(model, G))
        println("\nSaving $(normalise ? "normalised" : "") solution into $savepath...")

        jldsave(savepath; V̄, policy, model, G)
    end

    return V̄, policy, model, G
end