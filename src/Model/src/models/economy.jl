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
    ωᵣ::Float64 = 2e-3 # Speed of abatement technology cost reduction
    ϱ::Float64 = 9e-4 # Growth of TFP
    κ::Float64 = 12.3 # Adjustment costs of abatement technology
    δₖᵖ::Float64 = 0.0116 # Initial depreciation rate of capital

    # Output
    A₀::Float64 = 0.113 # Initial TFP
    Y₀::Float64 = 75.8
    σₖ::Float64 = 1.62e-2 # Variance of GDP

    τ::Float64 = 300. # Steady state horizon
end

"Cost of abatement as a fraction of GDP"
function β(t, e, economy::Economy)
    (e^2 / 2) * exp(-economy.ωᵣ * t)
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
    rate = (1 - χ) * A(t, economy)
    rate - (economy.κ / 2) * rate^2
end

function A(t, economy::Economy)
    economy.A₀ * exp(economy.ϱ * t)
end