Base.@kwdef struct GrowthDamages
    ξ::Float64 = 0.000075
    υ::Float64 = 3.25
end

Base.@kwdef struct LevelDamages
    ξ::Float64 = 0.00266
end

Damages = Union{GrowthDamages, LevelDamages}

Base.@kwdef struct Economy
    # Technology
    ωᵣ::Float64 = 0.017558043747351086 # Speed of abatement technology cost reduction
    ω₀::Float64 = 0.11 # Fraction of GDP required today to abate
    ϱ::Float64 = 1e-3 # Growth of TFP
    κ::Float64 = 11.2 # Adjustment costs of abatement technology
    δₖᵖ::Float64 = 0.0162 # Initial depreciation rate of capital

    # Output
    A₀::Float64 = 0.113 # Initial TFP
    Y₀::Float64 = 75.8
    σₖ::Float64 = 0.0162 # Variance of GDP

    τ::Float64 = 500. # Steady state horizon
end

"Cost of abatement as a fraction of GDP"
function β(t, e, economy::Economy)
    economy.ω₀ * exp(-economy.ωᵣ * t) * e^2
end

function β′(t, e, economy::Economy)
    exp(-economy.ωᵣ * t) * e
end

function d(T, damages::GrowthDamages, hogg::Hogg)
    damages.ξ * max(T - hogg.Tᵖ, 0.)^damages.υ
end

function d(T, damages::LevelDamages, hogg::Hogg)
    inv(1 + damages.ξ * max(T - hogg.Tᵖ, 0.)^2)
end

function ϕ(t, χ, economy::Economy)
    productivity = (1 - χ) * A(t, economy)
    adjcosts = economy.κ * productivity^2 / 2.
        
    productivity - adjcosts
end

function A(t, economy::Economy)
    economy.A₀ * exp(economy.ϱ * t)
end

function RegionalEconomies(kwargs...)
    economyhigh = Economy(
        Y₀ = 47.54,
        A₀ = 0.2, # Initial TFP
        ϱ = 5e-4,
        kwargs...
    )

    economylow = Economy(
        Y₀ = 28.25,
        A₀ = 0.05,
        ϱ = 1e-3,
        kwargs...
    )

    return (economyhigh, economylow)
end