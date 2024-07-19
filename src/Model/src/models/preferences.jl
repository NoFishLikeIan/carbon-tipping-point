Base.@kwdef struct LogUtility
    ρ::Float64 = 0.015  # Discount rate 
end

Base.@kwdef struct CRRA
    ρ::Float64 = 0.015  # Discount rate 
    θ::Float64 = 10.     # Relative risk aversion
end

Base.@kwdef struct LogSeparable
    ρ::Float64 = 0.015  # Discount rate 
    θ::Float64 = 7.     # Relative risk aversion
end

Base.@kwdef struct EpsteinZin
    ρ::Float64 = 0.015  # Discount rate 
    θ::Float64 = 10.    # Relative Risk Aversion
    ψ::Float64 = 1.5   # Elasticity of Intertemporal Complementarity 
end

Preferences = Union{CRRA, EpsteinZin, LogUtility}

function f(c, v, Δt, p::EpsteinZin)
    ψ⁻¹ = 1 / p.ψ
    aggregator = (1 - p.θ) / (1 - ψ⁻¹)

    β = exp(-p.ρ * Δt)

    consumption = Δt * c^(1 - ψ⁻¹)
    value = β * ((1 - p.θ) * v)^inv(aggregator)

    return ((consumption + value)^aggregator) / (1 - p.θ)
end

"""
Climate damage aggregator. `χ` is the consumtpion rate, `F′` is the expected value of `F` at `t + Δt` and `Δt` is the time step
"""
function g(χ, F′, Δt, p::EpsteinZin)
    ψ⁻¹ = inv(p.ψ)
    agg = (1 - p.θ) / (1 - ψ⁻¹)

    discounting = Δt / (1 + p.ρ * Δt)
    consumption = discounting * χ^(1 - ψ⁻¹)

    β = exp(-p.ρ * Δt)
    value = (F′)^inv(agg)

    return ((1 -  β) * consumption + β * value)^agg
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