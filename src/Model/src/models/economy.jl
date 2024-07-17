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
    κ::Float64 = 0.0632 # Adjustment costs of abatement technology
    δₖᵖ::Float64 = 0.0116 # Initial depreciation rate of capital

    # Output
    A₀::Float64 = 0.113 # Initial TFP
    Y₀::Float64 = 75.8
    σₖ::Float64 = 1.62e-2 # Variance of GDP

    # Domain 
    t₀::Float64 = -15. # Initial time of IPCC report
    t₁::Float64 = 80. # Horizon of IPCC report
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
    (1 - χ) * A(t, economy) - (economy.κ / 2) * (1 - χ)^2
end

function A(t, economy::Economy)
    economy.A₀ * exp(economy.ϱ * t)
end

# TODO: Move t₀, Economy -> Calibration
"Parametric form of γ: (t₀, ∞) → [0, 1]"
γ(t, economy::Economy, calibration::Calibration) = γ(t, calibration.γparameters, economy.t₀)
function γ(t, p, t₀)
   max(p[1] + p[2] * (t - t₀) + p[3] * (t - t₀)^2, 0.)
end

"Linear interpolation of emissions in `calibration`"
Eᵇ(t, economy::Economy, calibration::Calibration) = Eᵇ(t, economy.t₀, economy.t₁, calibration.emissions)
function Eᵇ(t, t₀, t₁, emissions)
    if t ≤ t₀ return first(emissions) end
    if t ≥ t₁ return last(emissions) end

    partition = range(t₀, t₁; length = length(emissions))
    udx = findfirst(tᵢ -> tᵢ > t, partition)
    ldx = udx - 1

    α = (t - partition[ldx]) / step(partition)

    return (1 - α) * emissions[ldx] + α * emissions[udx]
end