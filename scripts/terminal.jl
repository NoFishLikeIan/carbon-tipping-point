using Model, Grid

using JLD2, DotEnv
using UnPack: @unpack
using Polyester: @batch
using FastClosures: @closure
using Roots: find_zero
using Printf: @printf, @sprintf

include("utils/saving.jl")

function cost(Kᵢ, Tᵢ, Δt, χ, model::AbstractModel{GrowthDamages, P}) where P
    δ = outputdiscount(Tᵢ, Δt, χ, model)

    g(χ, δ * Kᵢ, Δt, model.preferences)
end

function cost(Kᵢ, Tᵢ, Δt, χ, model::AbstractModel{LevelDamages, P}) where P
    δ = outputdiscount(Tᵢ, Δt, χ, model)
    damage = d(Tᵢ, model.damages, model.hogg)

    g(damage * χ, δ * Kᵢ, Δt, model.preferences)
end

""
function steadystatestep!(K, policy, model::AbstractModel, G; indices = CartesianIndices(K), Δt = 0.05)    
    for idx in indices
        Kᵢ = K[idx]
        Tᵢ = G.X[idx].T
        objective = @closure χ -> cost(Kᵢ, Tᵢ, Δt, χ, model)

        Kᵢ′, χ = gssmin(objective, 0., 1.)
        K[idx] = Kᵢ′
        policy[idx] = χ
    end
end

function vfi(K₀, model, G; tol = 1e-3, maxiter = 10_000, verbose = false, indices = CartesianIndices(K₀), alternate = false, Δt = 0.05)
    Kᵢ = copy(K₀)
    Kᵢ₊₁ = copy(K₀)

    pᵢ = similar(Kᵢ)
    pᵢ₊₁ = similar(Kᵢ)

    verbose && println("Starting iterations...")
    for iter in 1:maxiter
        iterindices = ifelse(alternate && isodd(iter), indices, reverse(indices))

        steadystatestep!(Kᵢ₊₁, pᵢ₊₁, model, G; indices = iterindices, Δt = Δt)

        ε = maximum(abs.((Kᵢ₊₁ .- Kᵢ) ./ Kᵢ))
        α = maximum(abs.((pᵢ₊₁ .- pᵢ) ./ pᵢ))

        if verbose && (!alternate || isodd(iter))
            @printf("Iteration %i / %i, ε = %.8f and α = %.8f...\r", iter, maxiter, ε, α)
        end

        if ε < tol
            verbose && @printf("Converged in %i iterations, ε = %.8f and α = %.8f.\n", iter, ε, α)
            return Kᵢ₊₁, pᵢ₊₁
        end
        
        Kᵢ .= Kᵢ₊₁
        pᵢ .= pᵢ₊₁
    end

    @warn "Convergence failed."
    return Kᵢ₊₁, pᵢ₊₁
end

function computeterminal(model, G; verbose = true, withsave = true, datapath = "data", iterkwargs...)    

    K₀ = ones(size(G))
    K, policy = vfi(K₀, model, G; verbose, iterkwargs...)
    
    if withsave
    
        filename = makefilename(model, G)
        savepath = joinpath(datapath, "terminal", filename)
        println("Saving solution into $savepath...")
        jldsave(savepath; K, policy)
    end

    return K, policy
end