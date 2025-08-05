abstract type Damages{T <: Real} end

Base.@kwdef struct GrowthDamages{T} <: Damages{T}
    ξ₁::T = 0.0357 # Linear term in damage function
    ξ₂::T = 0.0018 # Quadratic term in damage function
end

Base.@kwdef struct LevelDamages{T} <: Damages{T}
    ξ::T = 0.00266
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

function d(T, m, damages::GrowthDamages, hogg::Hogg, feedback::Feedback)
    ΔT = max(T - hogg.Tᵖ, 0.)
    return (damages.ξ₁ + damages.ξ₂ * ΔT) * μ(T, m, hogg, feedback) / hogg.ϵ
end
function d(T, m, damages::GrowthDamages, hogg::Hogg)
    ΔT = max(T - hogg.Tᵖ, 0.)
    return (damages.ξ₁ + damages.ξ₂ * ΔT) * μ(T, m, hogg) / hogg.ϵ
end
function d(T, damages::LevelDamages, hogg::Hogg)
    ΔT = max(T - hogg.Tᵖ, 0.)
    return inv(1 + damages.ξ * ΔT^2)
end

function D(ΔT, damages::GrowthDamages)
    damages.ξ₁ * ΔT + damages.ξ₂ * ΔT^2 / 2.
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