Base.@kwdef struct Economy
    # Preferences
    ρ::Float32 = 1.5f-2 # Discount rate 
    θ::Float32 = 10f0 # Relative risk aversion
    ψ::Float32 = 1.5f0 # Elasticity of intertemporal substitution 

    # Technology
    ωᵣ::Float32 = 2f-3 # Speed of abatement technology cost reduction
    ϱ::Float32 = 8f-3 # Growth of TFP
    κ::Float32 = 3.72f-1 # Adjustment costs of abatement technology
    
    # Damages
    δₖᵖ::Float32 = 1.5f-2 # Initial depreciation rate of capital
    ξ::Float32 = 7.5f-5
    υ::Float32 = 3.25f0

    # Output
    A₀::Float32 = 0.113f0 # Initial TFP
    Y₀::Float32 = 75.8f0
    σₖ::Float32 = 1.62f-2 # Variance of GDP

    # Domain 
    t₀::Float32 = -15f0 # Initial time of IPCC report
    t₁::Float32 = 80f0 # Horizon of IPCC report
    τ::Float32 = 250f0 # Steady state horizon

    Y̲::Float32 = 0.9f0 * 75.8f0
    Ȳ::Float32 = 1.3f0 * 75.8f0
end

"Epstein-Zin aggregator"
function f(χ, y, u, economy::Economy)
    @unpack ρ, θ, ψ = economy

    δu = max(0f0, (1 - θ) * u)

    c = χ * exp(y)
    R = (c / δu^inv(1 - θ))^(1 - 1 / ψ)

    return ρ * δu / (1 - 1 / ψ) * (R - 1)
end
function Y∂f(χ, y, u, economy::Economy)
    @unpack ρ, θ, ψ = economy
    δu = max(0f0, (1 - θ) * u)

    c = χ * exp(y)
    R = (c / δu^inv(1 - θ))^(1 - 1 / ψ)

    ρ * δu * R / χ
end

"Computes f, Y∂f, and Y²∂²f without recomputing factors"
function epsteinzinsystem(χ, y, u, economy::Economy)
    @unpack ρ, θ, ψ = economy

    δu = max(0f0, (1 - θ) * u)

    c = χ * exp(y)
    R = (c / δu^inv(1 - θ))^(1 - 1 / ψ)

    f₀ = ρ * δu / (1 - 1 / ψ) * (R - 1)
    Yf₁ = ρ * δu * R / χ
    Y²f₂ = -ρ * δu * R / (χ^2 * ψ)

    return f₀, Yf₁, Y²f₂
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
    (1 - χ) * A(t, economy) - (economy.κ / 2f0) * (1 - χ)^2 * A(t, economy)^2 
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