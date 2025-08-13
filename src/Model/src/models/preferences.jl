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

function Preferences(; ρ = 0.015, θ = 10., ψ = 1.)
    if ψ ≈ θ ≈ 1.
        LogUtility(ρ)
    elseif ψ ≈ 1.
        LogSeparable(ρ, θ)
    elseif ψ ≈ 1 / θ
        CRRA(ρ, θ)
    else
        EpsteinZin(ρ, θ, ψ)
    end
end

discount(ρ, Δt) = inv(1 + ρ * Δt)

"Climate damage aggregator. `χ` is the consumtpion rate, `F′` is the expected value of `F` at `t + Δt` and `Δt` is the time step"
function g(χ, F′, Δt, p::EpsteinZin)
    β = discount(p.ρ, Δt)
    ψ⁻¹ = inv(p.ψ)
    agg = (1 - ψ⁻¹) / (1 - p.θ)

    consumption = χ^(1 - ψ⁻¹)
    value = (F′)^agg

    return ((1 -  β) * consumption + β * value)^inv(agg)
end
function g(χ, F′, Δt, p::LogSeparable)
    β = discount(p.ρ, Δt)
    consumption = χ^((1 - p.θ) * (1 - β))

    return consumption * max(F′, eps(F′))^β
end
function g(χ, F′, Δt, p::CRRA)
    β = discount(p.ρ, Δt)

    return (1 - β) * χ^(1 - p.θ) + β * F′
end
function g(χ, F′, Δt, p::LogUtility)
   β = discount(p.ρ, Δt)
   
   return (1 - β) * log(χ) + β * F′
end

function logg(χ, logF′, Δt, p::LogSeparable)
    β = discount(p.ρ, Δt)

    return (1 - p.θ) * (1 - β) * log(χ) + β * logF′
end


"Epstein-Zin aggregator"
function f(c, v, Δt, p::EpsteinZin)
    ψ⁻¹ = 1 / p.ψ
    aggregator = (1 - p.θ) / (1 - ψ⁻¹)

    β = discount(p.ρ, Δt)

    consumption = Δt * c^(1 - ψ⁻¹)
    value = β * ((1 - p.θ) * v)^inv(aggregator)

    return ((consumption + value)^aggregator) / (1 - p.θ)
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

Base.broadcastable(p::Preferences) = Ref(p)