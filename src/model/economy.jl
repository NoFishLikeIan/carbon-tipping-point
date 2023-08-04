using UnPack

include("climate.jl")

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
    Y₀::Float32 = 75.8f0 # trillion US-$
    A₀::Float32 = 0.113f0 # Initial TFP
end

"""
Parametric form of γ(t)
"""
function γ(t, p, t0)
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

    # FIXME check for ψ ≠ 1
    ψ⁻¹ = 1 / ψ

    return ρ * (1 - θ) * inv(1 - ψ⁻¹) * u * (
        (c / ((1 - θ) * u)^inv(1 - θ))^(1 - ψ⁻¹) - 1
    )
end


function ∂f_∂c(c, u, economy::Economy)
    @unpack ρ, θ, ψ = economy

    if ψ ≈ 1
        return ρ * (1 - θ) * (u / c)
    end

    ψ⁻¹ = 1 / ψ
    ucont = ((1 - θ) * u)^((1 - θ) / (1 - ψ⁻¹))

    return ρ * (1 - θ) * u * (c^(-ψ⁻¹) / ucont)
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

function δₖ(T, economy::Economy, baseline::Hogg)
    @unpack δₖᵖ = economy

    return δₖᵖ + (1 - δₖᵖ) * d(T, economy, baseline)
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
