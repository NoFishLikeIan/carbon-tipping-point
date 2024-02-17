using Model, Grid

using JLD2, DotEnv
using UnPack: @unpack
using Polyester: @batch
using FastClosures: @closure
using Roots: find_zero
using Printf: @printf, @sprintf

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
    
        dVT = G.h * abs(dT) * ifelse(dT > 0, VᵢT₊, VᵢT₋) + σₜ² * (VᵢT₊ + VᵢT₋) / 2
        
        value = @closure χ -> begin
            dy = bterminal(Xᵢ, χ, model) / G.Δ.y
            dVy = G.h * abs(dy) * ifelse(dy > 0, Vᵢy₊, Vᵢy₋) + σₖ² * (Vᵢy₊ + Vᵢy₋) / 2
            
            # Expected value
            Q = σₜ² + σₖ² + G.h * (abs(dT) + abs(dy))
            v = (dVy + dVT) / Q
            Δt = G.h^2 / Q

            c = χ * exp(Xᵢ.y)

            f(c, v, Δt, model.preferences)
        end

        # Optimal control
        v, χ = gss(value, 0., 1.)
        
        policy[idx] = χ
        V̄[idx] = v
    end

    return V̄, policy
end

function vfi(V₀::AbstractArray{Float64, 3}, model::ModelInstance, G::RegularGrid; tol = 1e-3, maxiter = 10_000, verbose = false, indices = CartesianIndices(G), alternate = false)
    pᵢ = 0.5 .* ones(size(G))
    pᵢ₊₁ = copy(pᵢ)

    Vᵢ = copy(V₀)
    Vᵢ₊₁ = copy(V₀)

    verbose && println("Starting iterations...")
    for iter in 1:maxiter
        iterindices = (alternate && isodd(iter)) ? indices : reverse(indices)

        terminaljacobi!(Vᵢ₊₁, pᵢ₊₁, model, G; indices = iterindices)

        ε = maximum(abs.((Vᵢ₊₁ .- Vᵢ) ./ Vᵢ))
        α = maximum(abs.(pᵢ₊₁ .- pᵢ) ./ pᵢ)

        verbose && @printf("Iteration %i / %i, ε = %.5f and α = %.5f...\r", iter, maxiter, ε, α)

        if ε < tol
            verbose && @printf("Converged in %i iterations, ε = %.5f and α = %.5f.\n", iter, ε, α)
            return Vᵢ₊₁, pᵢ₊₁
        end
        
        Vᵢ .= Vᵢ₊₁
        pᵢ .= pᵢ₊₁
    end

    @warn "Convergence failed."
    return Vᵢ₊₁, pᵢ₊₁
end

function computeterminal(model::ModelInstance, G::RegularGrid; verbose = true, withsave = true, datapath = "data", iterkwargs...)    
    Vcurve = [log(exp(Xᵢ.y)) / preferences.ρ for Xᵢ ∈ G.X]

    V₀ = typeof(preferences) <: EpsteinZin ? 
        Vcurve .- 2maximum(Vcurve) : Vcurve # Ensures V₀ < 0, for Epstein-Zin utilities

    V̄, policy = vfi(V₀, model, G; verbose, iterkwargs...)
    
    if withsave
        savepath = joinpath(datapath, "terminal", makefilename(model, G))
        println("Saving solution into $savepath...")
        jldsave(savepath; V̄, policy)
    end

    return V̄, policy, G
end