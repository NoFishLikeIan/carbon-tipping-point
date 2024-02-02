using Model, Grid

using JLD2, DotEnv
using UnPack: @unpack
using Polyester: @batch
using FastClosures: @closure
using Roots: find_zero

include("utils/saving.jl")

"Computes the Jacobi iteration for the terminal problem, V̄."
function terminaljacobi!(
    V̄::AbstractArray{Float64, 3}, 
    policy::AbstractArray{Float64, 3}, 
    model::ModelInstance, G::RegularGrid;
    indices = CartesianIndices(G)) # Can use internal indices to keep boundary constant

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

        value = @closure χ -> begin
            dy = bterminal(Xᵢ, χ, model) / G.Δ.y
            Q = σₜ² + σₖ² + G.h * (abs(dT) + abs(dy))

            py₊ = (G.h * max(dy, 0.) + σₖ² / 2) / Q
            py₋ = (G.h * max(-dy, 0.) + σₖ² / 2) / Q

            pT₊ = (G.h * max(dT, 0.) + σₜ² / 2) / Q
            pT₋ = (G.h * max(-dT, 0.) + σₜ² / 2) / Q

            # Expected value
            EV = py₊ * Vᵢy₊ + py₋ * Vᵢy₋ + pT₊ * VᵢT₊ + pT₋ * VᵢT₋

            Δt = G.h^2 / Q

            f(χ, Xᵢ, EV, Δt, model.preferences)
        end

        # Optimal control
        v, χ = gss(value, 0., 1.)
        
        V̄[idx] = v
        policy[idx] = χ
    end

    return V̄, policy
end

function terminaliteration(V₀::AbstractArray{Float64, 3}, model::ModelInstance, grid::RegularGrid; tol = 1e-3, maxiter = 100_000, verbose = false, indices = CartesianIndices(grid), alternate = false)
    policy = similar(V₀);

    verbose && println("Starting iterations...")

    Vᵢ, Vᵢ₊₁ = copy(V₀), copy(V₀);
    for iter in 1:maxiter
        terminaljacobi!(Vᵢ₊₁, policy, model, grid; indices = (alternate && isodd(iter)) ? reverse(indices) : indices)

        ε = maximum( abs.(Vᵢ₊₁ .- Vᵢ) )

        if verbose && isodd(iter)
            print("Iteration $iter / $maxiter, ε = $ε...\r")
        end

        if ε < tol
            return Vᵢ₊₁, policy
        end
        
        Vᵢ .= Vᵢ₊₁
    end

    verbose && println("\nDone without convergence.")
    return Vᵢ₊₁, policy
end

function computeterminal(N::Int, Δλ, preferences::Preferences; verbose = true, withsave = true, datapath = "data", iterkwargs...)
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
    
    V₀ = typeof(preferences) <: EpsteinZin ? 
        -ones(size(G)) : [log(exp(Xᵢ.y)) / preferences.ρ for Xᵢ ∈ G.X];

    V̄, policy = terminaliteration(V₀, model, G; verbose, iterkwargs...)

    if withsave
        savepath = joinpath(datapath, "terminal", filename(model, G))
        println("\nSaving solution into $savepath...")

        jldsave(savepath; V̄, policy, model, G)
    end

    return V̄, policy, model
end