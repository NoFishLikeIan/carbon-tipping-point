Base.@kwdef struct Economy
    # Preferences
    ρ::Float32 = 1.5f-3 # Discount rate 
    θ::Float32 = 10f0 # Relative risk aversion
    ψ::Float32 = 1.5f0 # Elasticity of intertemporal substitution 

    # Technology
    ωᵣ::Float32 = 2f-3 # Speed of abatement technology cost reduction
    ϱ::Float32 = 8f-3 # Growth of TFP
    κ::Float32 = 3.72f-1 # Adjustment costs of abatement technology
    
    # Damages
    δₖᵖ::Float32 = 1.5f-2 # Initial depreciation rate of capital
    damagehalftime::Float32 = 1.04f0 # d(T) = 1/2 if T = damagehalftime Tᵖ 

    # Output
    A₀::Float32 = 0.113f0 # Initial TFP
    Y₀::Float32 = 75.8f0

    # Domain 
    t₀::Float32 = -15f0 # Initial time of IPCC report
    t₁::Float32 = 80f0 # Horizon of IPCC report

    Y̲::Float32 = 0.9f0 * 75.8f0 # Current output is assumed to be minimum
    Ȳ::Float32 = 1.3f0 * 75.8f0 # Maximum output

    V̲::Float32 = 10f3
end

"Epstein-Zin aggregator"
function f(χ, y, u, economy::Economy)
    @unpack ρ, θ, ψ = economy

    c = χ * exp(y)
    R = (c / ((1 - θ) * u)^inv(1 - θ))^(1 - 1 / ψ)

    return ρ * (1 - θ) * u / (1 - 1 / ψ) * (R - 1)
end

function Y∂f(χ, y, u, economy::Economy)
    @unpack ρ, θ, ψ = economy

    R = (χ * exp(y) / ((1 - θ) * u)^inv(1 - θ))^(1 - 1 / ψ)

    return ρ * (1 - θ) * u * R / χ
end

function Y²∂²f(χ, y, u, economy::Economy)
    @unpack ρ, θ, ψ = economy

    R = (χ * exp(y) / ((1 - θ) * u)^inv(1 - θ))^(1 - 1 / ψ)

    return -ρ * (1 - θ) * u * R / (χ^2 * ψ)
end

"Cost of abatement as a fraction of GDP"
function β(t, e, economy::Economy)
    (e^2 / 2) * exp(-economy.ωᵣ * t)
end

function β′(t, e, economy::Economy)
    exp(-economy.ωᵣ * t) * e
end

function d(T, economy::Economy, hogg::Hogg)
    fct = 1 - (1 / economy.damagehalftime)
    ΔT = fct * hogg.Tᵖ + (1 - fct) * (hogg.Tᵖ - T)

    return inv(1 + exp(ΔT / 3))
end

function δₖ(T, economy::Economy, hogg::Hogg)
    @unpack δₖᵖ = economy

    return δₖᵖ + (1 - δₖᵖ) * d(T, economy, hogg)
end

function ϕ(t, χ, economy::Economy)
    (1 - χ) * A(t, economy) * (1 - (1 - χ) * A(t, economy) * economy.κ / 2)
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
    if t < t₀ return first(emissions) end
    if t > t₁ return last(emissions) end

    partition = range(t₀, t₁; length = length(emissions))
    udx = findfirst(tᵢ -> tᵢ > t, partition)
    ldx = udx - 1

    α = (t - partition[ldx]) / step(partition)

    return (1 - α) * emissions[ldx] + α * emissions[udx]
end

"Emissivity rate implied by abatement `α` at time `t` and carbon concentration `M`"
function ε(t, M::Real, α::Real, instance::ModelInstance, calibration::Calibration)
    economy, hogg, _ = instance

    1f0 - M * (δₘ(M, hogg) + γ(t, economy, calibration) - α) / (Gtonoverppm * Eᵇ(t, economy, calibration))
end
function ε(t, M::AbstractArray, α::Real, instance::ModelInstance, calibration::Calibration)
    economy, hogg, _ = instance
    1f0 .- M .* (δₘ.(M, Ref(hogg)) .+ γ(t, economy, calibration) .- α) ./ (Gtonoverppm * Eᵇ(t, economy, calibration))
end

function ε′(t, M::Real, instance::ModelInstance, calibration::Calibration)
    M / (Gtonoverppm * Eᵇ(t, instance[1], calibration))
end