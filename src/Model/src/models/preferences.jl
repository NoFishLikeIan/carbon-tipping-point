abstract type Preferences{T <: Real} end

Base.@kwdef struct LogUtility{T} <: Preferences{T}
    ρ::T = 0.015 # Discount rate
end

Base.@kwdef struct CRRA{T} <: Preferences{T}
    ρ::T = 0.015  # Discount rate 
    θ::T = 10.  # Relative risk aversion
end

Base.@kwdef struct LogSeparable{T} <: Preferences{T}
    ρ::T = 0.015  # Discount rate 
    θ::T = 10.  # Relative risk aversion
end

Base.@kwdef struct EpsteinZin{T} <: Preferences{T}
    ρ::T = 0.015 # Discount rate
    θ::T = 10. # Relative Risk Aversion
    ψ::T = 0.75 # Elasticity of Intertemporal Complementarity
end

Base.broadcastable(p::Preferences) = Ref(p)

function f(c, v, Δt, p::EpsteinZin)
    ψ⁻¹ = 1 / p.ψ
    aggregator = (1 - p.θ) / (1 - ψ⁻¹)

    β = exp(-p.ρ * Δt)

    consumption = Δt * c^(1 - ψ⁻¹)
    value = β * ((1 - p.θ) * v)^inv(aggregator)

    return ((consumption + value)^aggregator) / (1 - p.θ)
end

"Climate damage aggregator. `χ` is the consumtpion rate, `F′` is the expected value of `F` at `t + Δt` and `Δt` is the time step"
function g(χ, F′, Δt, p::EpsteinZin)
    ψ⁻¹ = inv(p.ψ)
    agg = (1 - ψ⁻¹) / (1 - p.θ)

    consumption = χ^(1 - ψ⁻¹)

    β = exp(-p.ρ * Δt)
    value = max(F′, 0.)^agg

    return ((1 -  β) * consumption + β * value)^inv(agg)
end

function logg(χ, F′, Δt, p::EpsteinZin)
    ψ⁻¹ = inv(p.ψ)
    agg = (1 - ψ⁻¹) / (1 - p.θ)

    consumption = χ^(1 - ψ⁻¹)

    β = exp(-p.ρ * Δt)
    value = max(F′, 0.)^agg

    return inv(agg) * log((1 -  β) * consumption + β * value)
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

function f(c, v, p::EpsteinZin)
    u = (1 - p.θ) * v
    eis = 1 - 1 / p.ψ

    cratio = (c / (u)^inv(1 - p.θ))^eis

    return (p.ρ * u / eis) * (cratio - 1) 
end