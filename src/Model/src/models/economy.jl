abstract type Damages{T <: Real} end
abstract type GrowthDamages{T} <: Damages{T} end
abstract type LevelDamages{T} <: Damages{T} end

Base.@kwdef struct WeitzmanGrowth{T} <: GrowthDamages{T}
    ξ::T = 2.6e-4
    ν::T = 3.25
end

Base.@kwdef struct Kalkuhl{T} <: GrowthDamages{T}
    ξ₁::T = 0.0357 # Linear term in damage function
    ξ₂::T = 0.0018 # Quadratic term in damage function
end

Base.@kwdef struct WeitzmanLevel{T} <: Damages{T}
    ξ::T = 0.00266
end

function d(T, m, damages::Kalkuhl, hogg::Hogg, feedback::Feedback)
    ΔT = max(T - hogg.Tᵖ, 0)
    return (damages.ξ₁ + damages.ξ₂ * ΔT) * μ(T, m, hogg, feedback) / hogg.ϵ
end
function d(T, m, damages::Kalkuhl, hogg::Hogg)
    ΔT = max(T - hogg.Tᵖ, 0)
    return (damages.ξ₁ + damages.ξ₂ * ΔT) * μ(T, m, hogg) / hogg.ϵ
end

function d(T, m, damages::WeitzmanGrowth, hogg::Hogg)
    ΔT = max(T - hogg.Tᵖ, 0)
    return damages.ξ * ΔT^damages.ν
end
function d(T, m, damages::WeitzmanGrowth, hogg::Hogg, feedback::Feedback)
    d(T, m, damages, hogg)
end

function d(T, m, damages::D, hogg::Hogg) where D <: LevelDamages
    d(T, damages, hogg)
end
function d(T, damages::WeitzmanLevel, hogg::Hogg)
    ΔT = max(T - hogg.Tᵖ, 0)
    return inv(1 + damages.ξ * ΔT^2)
end

function D(ΔT, damages::Kalkuhl)
    damages.ξ₁ * ΔT + damages.ξ₂ * ΔT^2 / 2.
end

Base.broadcastable(damages::Damages) = Ref(damages)

Base.@kwdef struct Economy{T <: Real}
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

    τ::T = 307. # Steady state horizon, such that `exp(-ϱ * τ) = 1%.`
end

"Cost of abatement as a fraction of GDP"
function β(t, e, economy::Economy)
    economy.ω₀ * exp(-economy.ωᵣ * t) * e^2 / 2.
end

function β′(t, e, economy::Economy)
    exp(-economy.ωᵣ * t) * e
end

function ϕ(t, χ, economy::Economy)
    productivity = (1 - χ) * A(t, economy)
    adjcosts = economy.κ * productivity^2 / 2.
        
    return productivity - adjcosts
end

function A(t, economy::Economy)
    economy.A₀ * exp(economy.ϱ * t)
end

function RegionalEconomies(kwargs...)
    economyhigh = Economy(
        Y₀ = 47.54,
        A₀ = 0.13, # Initial TFP
        ϱ = 0.000052,
        kwargs...
    )

    economylow = Economy(
        Y₀ = 28.25,
        A₀ = 0.09,
        ϱ = 0.004322045780109746,
        kwargs...
    )

    return (economyhigh, economylow)
end