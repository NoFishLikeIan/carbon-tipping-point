Gtonoverppm = 1 / 7.821

Base.@kwdef struct LinearQuadratic
    ρ::Float64 = 0.03 # Discount rate

    β₀::Float64 = 97.1 * Gtonoverppm # Linear benefit $ / ppm
    β₁::Float64 = 4.81 * Gtonoverppm^2 # Quadratic $ / (y ppm^2)

    γ::Float64 = 16.0 # Damage $ / K^2
    xₛ::Float64 = 287. # Surely safe temperature

    ē::Float64 = 200. # Maximum emissions
end

Base.@kwdef struct Ramsey
    ρ::Float64 = 0.024 # Discount rate

    α::Float64 = 0.4 # Capital share
    A::Float64 = 21.3 # Capital productivity

    β::Float64 = 22826.52702 # Emissions intensity

    γ₀::Float64 = 0.022 # Damage curvature
    γ₁::Float64 = 0.5 # Damage speed
    xₛ::Float64 = 287. # Surely safe temperature

    δₖ::Float64 = 0.0439 # Capital depreciation

    e₀::Float64 = 33 * Gtonoverppm # Initial emissions
    ē::Float64 = 20. # Maximum emissions

    L::Float64 = 3.43e9 # Labour supply
end

EconomicModel = Union{LinearQuadratic, Ramsey}

# Damage function
function d(x, economy::LinearQuadratic)
    @unpack xₛ, γ = economy
    return x ≥ xₛ ? (γ / 2) * (x - xₛ)^2 : 0.
end

function d(x, economy::Ramsey)
    @unpack xₛ, γ₁, γ₀ = economy
    Δx = x - xₛ

    return Δx > 0 ? 1 - exp(γ₀ * (1 - exp(γ₁ * Δx))) : 0.
end

# Consumption
function K(e, economy::Ramsey)
    @unpack α, A, L = economy
    return (Y(e, economy) / (A * L^(1 - α)))^(1 / α)
end

K′(e, economy::Ramsey) = K(e, economy) / (economy.α * e) 

Y(e, economy::Ramsey) = economy.β * e

function c(e, x, economy::Ramsey)
    (1 - d(x, economy)) * Y(e, economy) - economy.δₖ * K(e, economy)
end

function ∂ₑc(e, x, economy::Ramsey)
    if e ≈ 0 return Inf end
    (1 - d(x, economy)) * economy.β - economy.δₖ * K′(e, economy)
end

# Utility
function u(e, x, economy::LinearQuadratic)
    @unpack β₀, β₁ = economy
    β₀ * e - (β₁ / 2) * e^2 - d(x, economy)
end

function u(e, x, economy::Ramsey)
    consumption = c(e, x, economy)
    return consumption > 0 ? log(c(e, x, economy)) : -Inf
end