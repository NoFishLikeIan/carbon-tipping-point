"Emissivity rate implied by abatement `α` at time `t` and carbon concentration `M`"
function ε(t, M, α, instance::ModelInstance, calibration::Calibration)
    economy, hogg, _ = instance

    1f0 - M * (δₘ(M, hogg) + γ(t, economy, calibration) - α) / (Gtonoverppm * Eᵇ(t, economy, calibration))
end

function ε′(t, M, instance::ModelInstance, calibration::Calibration)
    M / (Gtonoverppm * Eᵇ(t, instance[1], calibration))
end

"Computes drift over whole state space `X`"
drift(t, X, policy::AbstractArray, instance::ModelInstance, calibration::Calibration) = drift!(similar(X), t, X, policy, instance, calibration)
function drift!(w, t, X, policy::AbstractArray, instance::ModelInstance, calibration::Calibration)
    economy, hogg, albedo = instance

    γₜ = γ(t, economy, calibration)
    Aₜ = A(t, economy)

    dimensions = size(X)[1:3]

    @batch for idx in CartesianIndices(dimensions)
        c = @view policy[idx, :]

        w[idx, 1] = μ(X[idx, 1], X[idx, 2], hogg, albedo)
        w[idx, 2] = γₜ - c[2]
        w[idx, 3] = economy.ϱ + ϕ(t, c[1], economy) -
            Aₜ *  β(t, ε(t, exp(X[idx, 2]), c[2], instance, calibration), economy) - 
            δₖ(X[idx, 1], economy, hogg)

    end

    return w
end

"""
Computes the Hamilton-Jacobi-Bellmann equation at point `Xᵢ`.
"""
function hjb(c, t, Xᵢ, Vᵢ, ∇Vᵢ, ∂²Vᵢ, instance::ModelInstance, calibration::Calibration)
    economy, hogg, albedo = instance

    f(c[1], Xᵢ[3], Vᵢ[1], economy) + 
        ∇Vᵢ[1] * μ(Xᵢ[1], Xᵢ[2], hogg, albedo) / hogg.ϵ +
        ∇Vᵢ[2] * (γ(t, economy, calibration) - c[2]) + 
        ∇Vᵢ[3] * (
            economy.ϱ + ϕ(t, c[1], economy) - economy.δₖᵖ - d(Xᵢ[1], economy, hogg) -
            A(t, economy) * β(t, ε(t, exp(Xᵢ[2]), c[2], instance, calibration), economy)
        ) +
        ∂²Vᵢ[1] * hogg.σ²ₜ / 2f0hogg.ϵ^2
end

"Constructs the negative objective functional at `Xᵢ`."
function objective!(Z, ∇, H, c, t, Xᵢ, Vᵢ, ∇Vᵢ, instance::ModelInstance, calibration::Calibration)
    economy = first(instance)

    f₀, Yf₁, Y²f₂ = epsteinzinsystem(c[1], Xᵢ[3], Vᵢ[1], economy)

    εₜ = ε(t, exp(Xᵢ[2]), c[2], instance, calibration)
    Aₜ = A(t, economy)

    if !isnothing(∇)
        ∇[1] = Yf₁ + ∇Vᵢ[3] * ϕ′(t, c[1], economy)
        ∇[2] = -∇Vᵢ[2] - ∇Vᵢ[3] * Aₜ * β′(t, εₜ, economy) * ε′(t, exp(Xᵢ[2]), instance, calibration)
    
        ∇ .*= -1f0
    end

    if !isnothing(H)
        H[1, 1] = Y²f₂ + ∇Vᵢ[3] * ϕ′′(t, economy)
        H[1, 2] = 0f0
        H[2, 1] = 0f0
        H[2, 2] = -∇Vᵢ[3] * Aₜ * ε′(t, exp(Xᵢ[2]), instance, calibration)^2  * exp(-economy.ωᵣ * t)
    
        H .*= -1f0
    end

    if !isnothing(Z)
        z = f₀ + 
            ∇Vᵢ[2] * (γ(t, economy, calibration) - c[2]) + 
            ∇Vᵢ[3] * (ϕ(t, c[1], economy) - Aₜ * β(t, εₜ, economy))

        return -z
    end

        
end