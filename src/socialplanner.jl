Base.@kwdef struct SocialPlanner
    η::Float64 = 1 / 2

    ρ::Float64 = 0.01
    τ::Float64 = 0.0
    d::Float64 = 1.0

    b::Float64 = 1.0
    a::Float64 = 1.0

    σ²::Float64 = 1.0
    Tᵣ::Float64 = 2.0
end

function F!(ddv, dv, v, p, t)

    η, ρ, τ, d, b, a, σ², Tᵣ = p

    ηᵣ = (1 - η) / η
    e = 1 / (ηᵣ * (τ + a * dv[1])^ηᵣ)

    ddv[1] = (ρ * v[1] + dv[1] * (t^2 - b) * t + d * (t - Tᵣ)^2 - e) / σ²
end

function E(λ, τ, a, η)
    inv(η) * inv(τ + a * λ)
end

function statecostate!(dx, x, p, t)
    λ, T = x
    η, ρ, τ, d, b, a, σ², Tᵣ = p

    dx[1] = λ * (ρ - 3T^2 + b) - d
    dx[2] = T^3 - b * T + E(λ, τ, a, η)
end