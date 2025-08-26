abstract type Damages{T<:Real} end
abstract type GrowthDamages{T} <: Damages{T} end

struct NoDamageGrowth{T} <: GrowthDamages{T} end

Base.@kwdef struct WeitzmanGrowth{T} <: GrowthDamages{T}
    ξ::T = 2.6e-4
    ν::T = 3.25
end

Base.@kwdef struct Kalkuhl{T} <: GrowthDamages{T}
    ξ₁::T = 0.0357 # Linear term in damage function
    ξ₂::T = 0.0018 # Quadratic term in damage function
end

abstract type LevelDamages{T} <: Damages{T} end

struct NoDamageLevel{T} <: LevelDamages{T} end

Base.@kwdef struct WeitzmanLevel{T} <: Damages{T}
    ξ::T = 0.00266
end

d(_, _, damages::NoDamageGrowth{T}, args...) where T = zero(T)

function d(T, m, damages::Kalkuhl, hogg::Hogg, feedback::Feedback)
    ΔT = max(T - hogg.Tᵖ, 0)
    dT = μ(T, m, hogg, feedback) / hogg.ϵ
    return (damages.ξ₁ + damages.ξ₂ * ΔT) * dT
end
function d(T, m, damages::Kalkuhl, hogg::Hogg)
    ΔT = max(T - hogg.Tᵖ, 0)
    dT = μ(T, m, hogg) / hogg.ϵ
    return (damages.ξ₁ + damages.ξ₂ * ΔT) * dT
end

function d(T, m, damages::WeitzmanGrowth, hogg::Hogg)
    ΔT = max(T - hogg.Tᵖ, 0)
    return damages.ξ * ΔT^damages.ν
end
function d(T, m, damages::WeitzmanGrowth, hogg::Hogg, feedback::Feedback)
    d(T, m, damages, hogg)
end

function d(T, m, damages::D, hogg::Hogg) where D<:LevelDamages
    d(T, damages, hogg)
end
function d(T, damages::WeitzmanLevel, hogg::Hogg)
    ΔT = max(T - hogg.Tᵖ, 0)
    return inv(1 + damages.ξ * ΔT^2)
end
d(_, _, damages::NoDamageLevel{T}, args...) where T = zero(T)

function D(ΔT, damages::Kalkuhl)
    damages.ξ₁ * ΔT + damages.ξ₂ * ΔT^2 / 2.
end

Base.broadcastable(damages::Damages) = Ref(damages)

Base.@kwdef struct Economy{T<:Real}
    # Technology
    ωᵣ::T = 0.017558043747351086 # Speed of abatement technology cost reduction
    ω₀::T = 2 * 0.11 # Fraction of GDP required today to abate
    ϱ::T = 1e-3 # Growth of TFP
    κ::T = 11.2 # Adjustment costs of abatement technology
    δₖᵖ::T = 0.0162 # Initial depreciation rate of capital

    # Output
    A₀::T = 0.113 # Initial TFP
    Y₀::T = 75.8
    σₖ::T = 0.0162 # Variance of GDP
end

function ω(t, economy::Economy)
    economy.ω₀ * exp(-economy.ωᵣ * t)
end

"Cost of abatement as a fraction of GDP"
function β(t, ε, economy::Economy)
    ω(t, economy) * ε^2 / 2.
end

function β′(t, ε, economy::Economy)
    ω(t, economy) * ε
end

function ϕ(t, χ, economy::Economy)
    productivity = (1 - χ) * A(t, economy)
    adjcosts = economy.κ * productivity^2 / 2.

    return productivity - adjcosts
end
function Φ(t, economy::Economy)
    Aₜ = A(t, economy)
    return clamp(1 - 1 / (economy.κ * Aₜ), 0, 1)
end

function A(t, economy::Economy)
    economy.A₀ * exp(economy.ϱ * t)
end

function RegionalEconomies(kwargs...)
    economyhigh = Economy(
        Y₀=47.54,
        A₀=0.13, # Initial TFP
        ϱ=0.000052,
        kwargs...
    )

    economylow = Economy(
        Y₀=28.25,
        A₀=0.09,
        ϱ=0.004322045780109746,
        kwargs...
    )

    return (economyhigh, economylow)
end