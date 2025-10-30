struct Abatement{S <: Real}
    ω̄::S # Fraction of GDP required long term to abate
    Δω::S # Additional fraction of GDP required to abate today
    ρ::S # Speed of abatement technology cost reduction
    b::S # Coefficient of abatement fraction
end

const HambelAbatement = Abatement{Float64}(4.3e-4, 5.506e-2, 1.48e-2, 2.8)

"Abatement tehcnology decay"
function ω(t, abatement::Abatement)
    @unpack ω̄, Δω, ρ = abatement

    return ω̄ + Δω * exp(-ρ * t)
end

"Cost of abatement as a fraction of GDP"
function β(t, ε, abatement::Abatement)
    ω(t, abatement) * ε^abatement.b
end
function β′(t, ε, abatement::Abatement)
    abatement.b * ω(t, abatement) * ε^(abatement.b - 1)
end

Base.@kwdef struct Investment{S <: Real}
    κ::S = 11.2 # Adjustment costs of abatement technology
    δₖᵖ::S = 0.0162 # Depreciation rate of capital
    σₖ::S = 0.0162 # Variance of GDP
    
    ϱ::S = 0. # Exogenous growth of TFP
    A₀::S = 0.113 # Initial TFP
end

function A(t, investments::Investment)
    investments.A₀ * exp(investments.ϱ * t)
end
function ϕ(t, χ, investments::Investment)
    investmentrate = (1 - χ) * A(t, investments)
    adjcosts = investments.κ * investmentrate^2 / 2.

    return investmentrate - adjcosts
end

# Growth Damages
abstract type Damages{S<:Real} end
abstract type GrowthDamages{S} <: Damages{S} end
struct NoDamageGrowth{S} <: GrowthDamages{S} end

d(_, _, damages::NoDamageGrowth{S}, args...) where S = zero(S)

Base.@kwdef struct WeitzmanGrowth{S} <: GrowthDamages{S}
    ξ::S = 2.6e-4
    ν::S = 3.25
end

d(T, _, damages::WeitzmanGrowth, _) = d(T, damages)
function d(T, damages::WeitzmanGrowth)
    damages.ξ * abs(T)^damages.ν
end
function d′(T, damages::WeitzmanGrowth)
    sign(T) * damages.ν * damages.ξ * abs(T)^(damages.ν - 1)
end

Base.@kwdef struct Kalkuhl{S} <: GrowthDamages{S}
    ξ₁::S = 0.0373 # [1/°C]
    ξ₂::S = 0.0018 # [1/°C²]
end

function d(T, m, damages::Kalkuhl, climate::C) where {C <: Climate}
    linear = (damages.ξ₁ + damages.ξ₂ * T) * μ(T, m, climate) / climate.hogg.ϵ
    quadratic = damages.ξ₂ * variance(T, climate.hogg)  

    return linear + quadratic
end

function D(T, damages::Kalkuhl)
    damages.ξ₁ * T + damages.ξ₂ * T^2 / 2.
end

"Quadratic temperature damages, as in Burket et al. (2016), with calibration by Kalkuhl & Wenz (2020)."
Base.@kwdef struct BurkeHsiangMiguel{S} <: GrowthDamages{S}
    ξ::S = 7.09e-4
end

d(T, _, damages::BurkeHsiangMiguel, _) = d(T, damages)
function d(T, damages::BurkeHsiangMiguel)
    damages.ξ * max(T, 0)^2
end

Base.broadcastable(damages::Damages) = Ref(damages)

Base.@kwdef struct Economy{S <: Real, D <: Damages{S}}
    Y₀::S = 75.8
    investments::Investment{S}
    abatement::Abatement{S}
    damages::D
end