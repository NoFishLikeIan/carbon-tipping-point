using Model, Grid

using JLD2, DotEnv
using UnPack: @unpack
using Polyester: @batch
using FastClosures: @closure
using Roots: find_zero
using Printf: @printf, @sprintf

include("utils/saving.jl")

"Computes the Jacobi iteration for the terminal problem, F̄."
function terminaljacobi!(F̄::AbstractMatrix{Float64}, policy::AbstractMatrix{Float64}, model::ModelInstance, G::RegularGrid; indices = CartesianIndices(G))

    L, R = extrema(CartesianIndices(G))
    
    for idx in indices
        @unpack θ, ρ, ψ = model.preferences
        σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * G.Δ.T))^2
        σₘ² = (model.hogg.σₘ / G.Δ.m)^2

        Xᵢ = G.X[idx]
        dT = μ(Xᵢ.T, Xᵢ.m, model.hogg, model.albedo) / (model.hogg.ϵ * G.Δ.T)
    
        # Neighbouring nodes
        # -- Temperature
        FᵢT₊ = F̄[min(idx + I[1], R)]
        FᵢT₋ = F̄[max(idx - I[1], L)]

        Fᵢm₊ = F̄[min(idx + I[2], R)]
        Fᵢm₋ = F̄[max(idx - I[2], L)]
    
        dTF = G.h * abs(dT) * ifelse(dT > 0, FᵢT₊, FᵢT₋)+ σₜ² * (FᵢT₊ + FᵢT₋) / 2
        dmF = σₘ² * (Fᵢm₊ + Fᵢm₋) / 2

        Q = σₘ² + σₜ² + G.h * abs(dT)
        F′ = (dTF + dmF) / Q

        σₖ² = model.economy.σₖ^2
        damage = d(Xᵢ.T, model.damages, model.hogg)
        growth = model.economy.ϱ - model.economy.δₖᵖ

        Δt = G.h^2 / Q
        
        costs = @closure χ -> begin
            investment = ϕ(model.economy.τ, χ, model.economy)
            μy = growth + investment - damage
            δy = max(1 + (1 - θ) * (μy - θ * σₖ²) * Δt, 0.)

            return g(χ, δy * F′, Δt, model.preferences)
        end

        # Optimal control
        objmin, χ = gssmin(costs, 0., 1.; tol = 1e-6)
        
        policy[idx] = χ
        F̄[idx] = objmin
    end

    return F̄, policy
end
function terminaljacobi!(F̄::AbstractMatrix{Float64}, policy::AbstractMatrix{Float64}, model::ModelBenchmark, G::RegularGrid; indices = CartesianIndices(G))

    L, R = extrema(CartesianIndices(G))
    
    for idx in indices
        @unpack θ, ρ, ψ = model.preferences
        σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * G.Δ.T))^2
        σₘ² = (model.hogg.σₘ / G.Δ.m)^2

        Xᵢ = G.X[idx]
        dT = μ(Xᵢ.T, Xᵢ.m, model.hogg) / (model.hogg.ϵ * G.Δ.T)

        # Jump
        πᵢ = intensity(Xᵢ.T, model.hogg, model.jump)
        qᵢ = increase(Xᵢ.T, model.hogg, model.jump)

        steps = floor(Int, div(qᵢ, G.Δ.T * G.h))
        weight = qᵢ / (G.Δ.T * G.h)

        Fʲ = F̄[min(idx + steps * I[1], R)] * (1 - weight) +
                F̄[min(idx + (steps + 1) * I[1], R)] * weight

        # Neighbouring nodes
        # -- Temperature
        FᵢT₊ = F̄[min(idx + I[1], R)]
        FᵢT₋ = F̄[max(idx - I[1], L)]

        Fᵢm₊ = F̄[min(idx + I[2], R)]
        Fᵢm₋ = F̄[max(idx - I[2], L)]
    
        dTF = G.h * abs(dT) * ifelse(dT > 0, FᵢT₊, FᵢT₋)+ σₜ² * (FᵢT₊ + FᵢT₋) / 2
        dmF = σₘ² * (Fᵢm₊ + Fᵢm₋) / 2

        Q = σₘ² + σₜ² + G.h * abs(dT)
        Δt = G.h^2 / Q
        Fᵈ = (dTF + dmF) / Q
        F′ = Fᵈ + πᵢ * Δt * (Fʲ - Fᵈ)

        σₖ² = model.economy.σₖ^2
        damage = d(Xᵢ.T, model.damages, model.hogg)
        growth = model.economy.ϱ - model.economy.δₖᵖ
        
        costs = @closure χ -> begin
            investment = ϕ(model.economy.τ, χ, model.economy)
            μy = growth + investment - damage
            δy = max(1 + (1 - θ) * (μy - θ * σₖ²) * Δt, 0.)

            return g(χ, δy * F′, Δt, model.preferences)
        end

        # Optimal control
        objmin, χ = gssmin(costs, 0., 1.; tol = 1e-6)
        
        policy[idx] = χ
        F̄[idx] = objmin
    end

    return F̄, policy
end

function vfi(F₀::AbstractMatrix{Float64}, model, G::RegularGrid; tol = 1e-3, maxiter = 10_000, verbose = false, indices = CartesianIndices(G), alternate = false)
    pᵢ = 0.5 .* ones(size(G))
    pᵢ₊₁ = copy(pᵢ)

    Fᵢ = copy(F₀)
    Fᵢ₊₁ = copy(F₀)

    verbose && println("Starting iterations...")
    for iter in 1:maxiter
        iterindices = (alternate && isodd(iter)) ? indices : reverse(indices)

        terminaljacobi!(Fᵢ₊₁, pᵢ₊₁, model, G; indices = iterindices)

        ε = maximum(abs.((Fᵢ₊₁ .- Fᵢ)))
        α = maximum(abs.(pᵢ₊₁ .- pᵢ))

        verbose && @printf("Iteration %i / %i, ε = %.8f and α = %.8f...\r", iter, maxiter, ε, α)

        if ε < tol
            verbose && @printf("Converged in %i iterations, ε = %.8f and α = %.8f.\n", iter, ε, α)
            return Fᵢ₊₁, pᵢ₊₁
        end
        
        Fᵢ .= Fᵢ₊₁
        pᵢ .= pᵢ₊₁
    end

    @warn "Convergence failed."
    return Fᵢ₊₁, pᵢ₊₁
end

function computeterminal(model, G::RegularGrid; verbose = true, withsave = true, datapath = "data", iterkwargs...)    

    V₀ = [log(exp(Xᵢ.y)) / preferences.ρ for Xᵢ ∈ G.X];
    V₀ = (V₀ .- 2maximum(V₀)) / 2maximum(V₀); # Ensurate that V₀ < 0

    V̄, policy = vfi(V₀, model, G; verbose, iterkwargs...)

    
    if withsave
        folder = typeof(model) <: ModelInstance ? "albedo" : "jump"
        filename = makefilename(model, G)
        savepath = joinpath(datapath, folder, "terminal", filename)
        println("Saving solution into $savepath...")
        jldsave(savepath; V̄, policy)
    end

    return V̄, policy, G
end