struct Abatement{S <: Real}
    ω̄::S # Fraction of GDP required long term to abate
    Δω::S # Additional fraction of GDP required to abate today
    ρ::S # Speed of abatement technology cost reduction
    b::S # Coefficient of abatement fraction
end
struct PiecewiseAbatement{S <: Real, A₁ <: Abatement{S}, A₂ <: Abatement{S}}
    abatement::A₁
    sequestration::A₂
end

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

function β(t, ε, piecewise::PiecewiseAbatement)
    β(t, min(ε, 1), piecewise.abatement) + β(t, max(ε - 1, 0), piecewise.sequestration)
end

function β′(t, ε, piecewise::PiecewiseAbatement)
    ε < 1 ? β′(t, ε, piecewise.abatement) :  β′(t, ε - 1, piecewise.sequestration)
end

Base.@kwdef struct Investment{S <: Real}
    κ::S = 11.2 # Adjustment costs of abatement technology
    δₖᵖ::S = 0.0162 # Initial depreciation rate of capital
    σₖ::S = 0.0162 # Variance of GDP
    
    ϱ::S = 1e-3 # Growth of TFP
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
Base.@kwdef struct WeitzmanGrowth{S} <: GrowthDamages{S}
    ξ::S = 2.6e-4
    ν::S = 3.25
end
Base.@kwdef struct Kalkuhl{S} <: GrowthDamages{S}
    ξ₁::S = 0.035 # [1/°C]
    ξ₂::S = 0.0018 # [1/°C²]
end

d(_, _, damages::NoDamageGrowth{S}, args...) where S = zero(S)
function d(T, m, damages::Kalkuhl, climate::C) where {C <: Climate}
    driftdamage = damages.ξ₁ * μ(T, m, climate) / climate.hogg.ϵ
    noisedamage = damages.ξ₂ * variance(T, climate.hogg) / 2
    return driftdamage + noisedamage
end
function d(T, _, damages::WeitzmanGrowth, climate::C) where {C <: Climate}
    return damages.ξ * T^damages.ν
end

function D(T, damages::Kalkuhl)
    damages.ξ₁ * T + damages.ξ₂ * T^2 / 2.
end

Base.broadcastable(damages::Damages) = Ref(damages)

Base.@kwdef struct Economy{S <: Real, D <: Damages{S}}
    Y₀::S = 75.8
    investments::Investment{S} = Investment{S}()
    abatement::Abatement{S}
    damages::D
end