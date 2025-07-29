Base.@kwdef struct GrowthDamages
    ξ₁::Float64 = 0.0357 # Linear term in damage function
    ξ₂::Float64 = 0.0018 # Quadratic term in damage function
end

Base.@kwdef struct LevelDamages
    ξ::Float64 = 0.00266
end

Damages = Union{GrowthDamages, LevelDamages}
Base.broadcastable(damages::Damages) = Ref(damages)

Base.@kwdef struct Economy
    # Technology
    ωᵣ::Float64 = 0.017558043747351086 # Speed of abatement technology cost reduction
    ω₀::Float64 = 2 * 0.11 # Fraction of GDP required today to abate
    ϱ::Float64 = 1e-3 # Growth of TFP
    κ::Float64 = 11.2 # Adjustment costs of abatement technology
    δₖᵖ::Float64 = 0.0162 # Initial depreciation rate of capital

    # Output
    A₀::Float64 = 0.113 # Initial TFP
    Y₀::Float64 = 75.8
    σₖ::Float64 = 0.0162 # Variance of GDP

    τ::Float64 = 307. # Steady state horizon, such that `exp(-ϱ * τ) = 1%.`
end

"Cost of abatement as a fraction of GDP"
function β(t, e, economy::Economy)
    economy.ω₀ * exp(-economy.ωᵣ * t) * e^2 / 2.
end

function β′(t, e, economy::Economy)
    exp(-economy.ωᵣ * t) * e
end

function d(T, m, damages::GrowthDamages, hogg::Hogg, albedo::Albedo)
    ΔT = max(T - hogg.Tᵖ, 0.)
    return (damages.ξ₁ + damages.ξ₂ * ΔT) * μ(T, m, hogg, albedo) / hogg.ϵ
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