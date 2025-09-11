const Gtonoverppm = 1 / 7.821

Base.@kwdef struct Feedback{S <: Real}
    Tᶜ::S # [K] Critical temperature in deviation from Tᵖ
    ΔS::S # [W m⁻²] Additional forcing
    L::S
end

# Feedback functions
function L(T, feedback::Feedback)
    logistic(feedback.L * (T - feedback.Tᶜ))
end
function L′(T, feedback::Feedback)
    l = L(T, feedback)
    return feedback.L * l * (1 - l)
end

function λ(T, feedback::Feedback)
    feedback.ΔS * L(T, feedback)
end
function λ′(T, feedback::Feedback)
    feedback.ΔS * L′(T, feedback) 
end

"Creates new Feedback object with updated critical temperature Tᶜ."
function updateTᶜ(Tᶜ, feedback::Feedback)
    Feedback(Tᶜ = Tᶜ, ΔS = feedback.ΔS, L = feedback.L)
end

Base.@kwdef struct Jump{S <: Real}
    j₂::S = -0.0029
    j₁::S = 0.0568
    j₀::S = -0.0577

    i₀::S = -0.25
    i₁::S = 0.95
    
    e₁::S = 2.8
    e₂::S = -0.3325
end

abstract type Decay{S <: Real} end
Base.@kwdef struct ConstantDecay{S} <: Decay{S}
    δ::S = 0.0176
end
struct ExponentialDecay{S} <: Decay{S}
    aδ::S
    bδ::S
    cδ::S
end

Base.@kwdef struct Hogg{S <: Real}
    # Defaults values
    T₀::S = 1.33 # [K] in deviation from Tᵖ
    Tᵖ::S = 287.15 # [K]
    M₀::S = 563.88 # [p.p.m. CO₂e]
    Mᵖ::S = 383.15 # [p.p.m. CO₂e]
    N₀::S = 388.38 # [p.p.m.]

    S₀::S = 235.0 # [W m⁻²] Mean solar radiation

    ϵ::S = 1.0440986343061285 # [yr J m⁻² K⁻¹] Speed of temperature
    η::S = 5.67e-8 # Stefan-Boltzmann constant 
    
    G₁::S = 22.685575408165214 # [W m⁻²] Effect of CO₂ on radiation budget
    G₀::S = 149.62958104691003 # [W m⁻²] Pre-industrial GHG radiation budget

    # Noise
    σₜ::S = 0.1441620917527504 # Standard deviation of temperature
    σₘ::S = 0.0078 # Standard deviation of CO₂e growth
end

"Approximation of CO₂e concentration decay."
function δₘ(_, decay::ConstantDecay)
    decay.δ
end
function δₘ(M, decay::ExponentialDecay)
    @unpack aδ, bδ, cδ = decay

    return aδ * exp(-((M - bδ) / cδ)^2)
end

abstract type Climate{S <: Real, D <: Decay{S}} end
abstract type PiecewiseLinearClimate{S <: Real, D <: Decay{S}} <: Climate{S, D} end
struct LinearClimate{S, D} <: PiecewiseLinearClimate{S, D}
    hogg::Hogg{S}
    decay::D
end
struct TippingClimate{S, D} <: Climate{S, D}
    hogg::Hogg{S}
    decay::D
    feedback::Feedback{S}
end
struct JumpingClimate{S, D} <: PiecewiseLinearClimate{S, D}
    hogg::Hogg{S}
    decay::D
    jump::Jump{S}
end

function Base.isless(a::C₁, b::C₂) where {C₁ <: TippingClimate, C₂ <: TippingClimate}
    isless(a.feedback.Tᶜ, b.feedback.Tᶜ)
end
function Base.isless(::C₁, ::C₂) where {C₁ <: PiecewiseLinearClimate, C₂ <: TippingClimate}
    false # Consistent with linear ≡ (Tᶜ → ∞)
end
function Base.isless(a::C₁, b::C₂) where {C₁ <: TippingClimate, C₂ <: PiecewiseLinearClimate}
    !isless(b, a)
end

"Forcing due to greenhouse gasses."
function ghgforcing(m, hogg::Hogg)
    hogg.G₀ + hogg.G₁ * m
end
function ghgforcing′(m, hogg::Hogg)
    hogg.G₁
end
function ghgforcinginverse(r, hogg::Hogg)
    (r - hogg.G₀) / hogg.G₁
end

"Forcing due to incoming solar radiation."
@fastpow function radiativeforcing(T, hogg::Hogg)
    hogg.S₀ - hogg.η * (T + hogg.Tᵖ)^4
end
@fastpow function radiativeforcing′(T, hogg::Hogg)
    -4hogg.η * (T + hogg.Tᵖ)^3
end

"Temperature drift."
function μ(T, m, climate::C) where {C <: PiecewiseLinearClimate}
    ghgforcing(m, climate.hogg) + radiativeforcing(T, climate.hogg)
end
function μ(T, m, climate::TippingClimate)
    ghgforcing(m, climate.hogg) + 
    radiativeforcing(T, climate.hogg) + 
    λ(T, climate.feedback)
end

function ∂μ∂T(T, climate::C) where {C <: PiecewiseLinearClimate}
    radiativeforcing′(T, climate.hogg)
end
function ∂μ∂T(T, climate::TippingClimate)
    radiativeforcing′(T, climate.hogg) + λ′(T, climate.feedback)
end
function ∂μ∂m(m, climate::C) where {C <: Climate}
    ghgforcing′(m, climate.hogg)
end

"CO₂e log-concentration consistent with temperature T."
function mstable(T, climate::C) where {C <: PiecewiseLinearClimate}
    r = radiativeforcing(T, climate.hogg)
    return ghgforcinginverse(-r, climate.hogg)
end
function mstable(T, climate::TippingClimate)
    r = radiativeforcing(T, climate.hogg) + λ(T, climate.feedback)
    return ghgforcinginverse(-r, climate.hogg)
end

"Temperature(s) consistent with CO₂e log-concentration m"
function Tstable(m, climate::C; Tmin = 0.8climate.hogg.Tᵖ, Tmax = 2climate.hogg.Tᵖ) where {C <: Climate}
    find_zeros(T -> mstable(T, climate) - m, Tmin, Tmax)
end

const log2 = log(2)

"Compute equilibrium climate sensitivity"
function ecs(climate::C) where {C <: Climate}
    only(Tstable(log2, climate)) - climate.hogg.Tᵖ
end

"Size of jump"
function increase(T, climate::C) where {C <: JumpingClimate}
    ΔT = T - climate.hogg.Tᵖ

    climate.jump.j₀ + climate.jump.j₁ * ΔT + climate.jump.j₂ * ΔT^2
end


"Arrival rate of jump"
function intensity(T, climate::C) where {C <: JumpingClimate}
    ΔT = T - climate.hogg.Tᵖ
    
    max(climate.jump.i₀ + climate.jump.i₁ / (1 + climate.jump.e₁ * exp(climate.jump.e₂ * ΔT)), 0)
end


function determnistichogg(hogg::Hogg{S}) where S
    Hogg{S}(
        T₀ = hogg.T₀,
        Tᵖ = hogg.Tᵖ,
        M₀ = hogg.M₀,
        Mᵖ = hogg.Mᵖ,
        N₀ = hogg.N₀,
        S₀ = hogg.S₀,
        ϵ = hogg.ϵ,
        η = hogg.η,
        G₁ = hogg.G₁,
        G₀ = hogg.G₀,
        σₜ = zero(S),
        σₘ = zero(S)
    )
end

function deterministicClimate(climate::LinearClimate)
    LinearClimate(determnistichogg(climate.hogg), climate.decay)
end

function deterministicClimate(climate::TippingClimate)
    TippingClimate(determnistichogg(climate.hogg), climate.decay, climate.feedback)
end

function deterministicClimate(climate::JumpingClimate)
    JumpingClimate(determnistichogg(climate.hogg), climate.decay, climate.jump)
end