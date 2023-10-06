using UnPack

Base.@kwdef struct Economy
    # Preferences
    ρ::Float32 = 1.5f-3 # Discount rate 
    θ::Float32 = 10f0 # Relative risk aversion
    ψ::Float32 = 1f0 # Elasticity of intertemporal substitution 

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

"""
Parametric form of γ: (t0, ∞) → [0, 1]
"""
function γ(t, p::NTuple{3, Float32}, t0)
    p[1] + p[2] * (t - t0) + p[3] * (t - t0)^2
end

"Epstein-Zin aggregator"
function f(c, u, economy::Economy)
    @unpack ρ, θ, ψ = economy

    if ψ ≈ 1
        return ρ * (1 - θ) * u * (
            log(
                c / ((1 - θ) * u)^inv(1 - θ)
            )
        )
    end

    @warn "Case for ψ ≠ 1 not implemented!"
end

function f(χ, y, u, economy::Economy)
    @unpack ρ, θ = economy
    ρ * min(u, -ϵ) * ((1 - θ) * (log(χ) + y) + log((1 - θ) * min(u, -ϵ)))
end

function ∂f_∂c(c, u, economy::Economy)
    @unpack ρ, θ, ψ = economy

    if ψ ≈ 1
        return ρ * (1 - θ) * (u / c)
    end

    @warn "Case for ψ ≠ 1 not implemented!"
end

"Cost of abatement as a fraction of GDP"
function β(t, ε, economy::Economy)
    return (ε^2 / 2) * exp(-economy.ωᵣ * t)
end


function d(T, economy::Economy, baseline::Hogg)
    fct = 1 - (1 / economy.damagehalftime)
    ΔT = fct * baseline.Tᵖ + (1 - fct) * (baseline.Tᵖ - T)

    return inv(1 + exp(ΔT / 3))
end

function δₖ(T, economy::Economy, hogg::Hogg)
    @unpack δₖᵖ = economy

    return δₖᵖ + (1 - δₖᵖ) * d(T, economy, hogg)
end

function ϕ(χ, A::Real, economy::Economy)
    I = (1 - χ) * A

    return I * (1 - I * economy.κ / 2)
end

function ϕ′(χ, A::Real, economy::Economy)
    return economy.κ * (1 - χ) * A^2 - A
end

function A(t, economy::Economy)
    return economy.A₀ * exp(economy.ϱ * t)
end
