gigatonco2toppm = 7.821

Base.@kwdef struct LinearQuadratic
    ρ::Float64 = 0.01 # Discount rate

    β₀::Float64 = 97.1 / gigatonco2toppm # Linear benefit $ / ppm
    β₁::Float64 = 4.81 / gigatonco2toppm^2 # Quadratic $ / (y ppm^2)

    τ::Float64 = 0.0 # Carbon tax $ / ppm

    γ::Float64 = 7.51443e-4 # Damage $ / K^2
    xₛ::Float64 = 289 # Surely safe temperature
end

function d(x, l::LinearQuadratic)
    (l.γ / 2) * (x - l.xₛ)^2
end

function d′(x, l::LinearQuadratic)
    l.γ * (x - l.xₛ)
end

"""
Utility of emissions given a carbon tax τ(x), which depends on temperature.
"""
function u(e, l::LinearQuadratic)
    (l.β₀ - l.τ) * e - (l.β₁ / 2) * e^2
end