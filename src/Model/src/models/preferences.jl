Base.@kwdef struct LogUtility
    ρ::Float64 = 0.015  # Discount rate 
end


Base.@kwdef struct CRRA
    ρ::Float64 = 0.015  # Discount rate 
    θ::Float64 = 7.    # Relative risk aversion
end

Base.@kwdef struct LogSeparable
    ρ::Float64 = 0.015  # Discount rate 
    θ::Float64 = 7.    # Relative risk aversion
end

Base.@kwdef struct EpsteinZin
    ρ::Float64 = 0.015  # Discount rate 
    θ::Float64 = 7.    # Relative risk aversion
    ψ::Float64 = 0.9    # Elasticity of intertemporal complementarity 
end

Preferences = Union{CRRA, EpsteinZin, LogUtility}

function f(c, v, Δt, p::EpsteinZin)
    ψ⁻¹ = 1 / p.ψ
    coeff = (1 - p.θ) / (1 - ψ⁻¹)

    βᵢ = exp(-p.ρ * Δt)

    consumption = (1 - βᵢ) * c^(1 - ψ⁻¹)
    value = βᵢ * ((1 - p.θ) * v)^inv(coeff)

    return ((consumption + value)^coeff) / (1 - p.θ)
end
function f(c, v, Δt, p::CRRA)
    u = (c^(1 - p.θ)) / (1 - p.θ)

    βᵢ = exp(-p.ρ * Δt)

    βᵢ * v + Δt * u
end
function f(c, v, Δt, p::LogUtility)
    βᵢ = exp(-p.ρ * Δt)

    βᵢ * v + Δt * log(c)
end