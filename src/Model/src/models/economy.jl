Base.@kwdef struct Economy
    # Technology
    ωᵣ::Float64 = 2e-3 # Speed of abatement technology cost reduction
    ϱ::Float64 = 9e-3 # Growth of TFP
    κ::Float64 = 0.372 # Adjustment costs of abatement technology
    
    # Damages
    δₖᵖ::Float64 = 1.5e-1 # Initial depreciation rate of capital
    ξ::Float64 = 0.00026
    υ::Float64 = 3.25

    # Output
    A₀::Float64 = 0.113 # Initial TFP
    Y₀::Float64 = 75.8
    σₖ::Float64 = 1.62e-2 # Variance of GDP

    # Domain 
    t₀::Float64 = -15. # Initial time of IPCC report
    t₁::Float64 = 80. # Horizon of IPCC report
    τ::Float64 = 80. # Steady state horizon
end

"Cost of abatement as a fraction of GDP"
function β(t, e, economy::Economy)
    (e^2 / 2) * exp(-economy.ωᵣ * t)
end

function β′(t, e, economy::Economy)
    exp(-economy.ωᵣ * t) * e
end

function d(T, economy::Economy, hogg::Hogg)
    @unpack υ, ξ = economy
    ξ * (T - hogg.Tᵖ)^υ
end

function d′(T, economy::Economy, hogg::Hogg)
    @unpack υ, ξ = economy
    ξ * υ * (T - hogg.Tᵖ)^(υ - 1)
end

function d′′(T, economy::Economy, hogg::Hogg)
    @unpack υ, ξ = economy
    ξ * υ * (υ - 1) * (T - hogg.Tᵖ)^(υ - 2)
end

function δₖ(T, economy::Economy, hogg::Hogg)
    economy.δₖᵖ + d(T, economy, hogg)
end

function ϕ(t, χ, economy::Economy)
    (1 - χ) * A(t, economy) - (economy.κ / 2) * (1 - χ)^2 * A(t, economy)^2 
end
function ϕ′(t, χ, economy::Economy)
    economy.κ * A(t, economy)^2 * (1 - χ) - A(t, economy)
end

function ϕ′′(t, economy::Economy)
    -economy.κ * A(t, economy)^2 
end

function A(t, economy::Economy)
    economy.A₀ * exp(economy.ϱ * t)
end

"Parametric form of γ: (t₀, ∞) → [0, 1]"
γ(t, economy::Economy, calibration::Calibration) = γ(t, calibration.γparameters, economy.t₀)
function γ(t, p, t₀)
    p[1] + p[2] * (t - t₀) + p[3] * (t - t₀)^2
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