Base.@kwdef struct Economy
    # Preferences
    ρ::Float64 = 0.015 # Discount rate 
    θ::Float64 = 10.0 # Relative risk aversion
    ψ::Float64 = 0.5 # Elasticity of intertemporal substitution 

    # Technology
    ωᵣ::Float64 = 0.02 # Speed of abatement technology cost reduction
    ϱ::Float64 = 0.015 # Growth of TFP
    κ::Float64 = 0.372 # Adjustment costs of abatement technology
    
    # Damages
    δₖᵖ::Float64 = 0.02 # Initial depreciation rate of capital
    damagehalftime::Float64 = 1.04 # d(T) = 1/2 if T = damagehalftime Tᵖ 

    # Output
    Y₀::Float64 = 75.8 # trillion US-$
    A₀::Float64 = 0.113 # Initial TFP
end

"Epstein-Zin aggregator"
function f(c, u, economy::Economy)
    @unpack ρ, θ, ψ = economy
    ψ⁻¹ = 1 / ψ

    ucont = ((1 - θ) * u)^((1 - θ) / (1 - ψ⁻¹))
    ccont = c^(1 - ψ⁻¹)

    return (ρ / (1 - ψ⁻¹)) * (1 - θ) * u * ((ccont / ucont) - 1)
end

function ∂cf(c, u, economy::Economy)
    @unpack ρ, θ, ψ = economy
    ψ⁻¹ = 1 / ψ

    ucont = ((1 - θ) * u)^((1 - θ) / (1 - ψ⁻¹))

    return ρ * (1 - θ) * u * c^(-ψ⁻¹) / ucont
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

function ϕ(χ, A, economy::Economy)
    I = (1 - χ) * A

    return I - (economy.κ / 2) * I^2
end

function A(t, economy::Economy)
    return economy.A₀ * exp(economy.ϱ * t)
end