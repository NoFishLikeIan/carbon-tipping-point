const Gtonoverppm = 1 / 7.821

Base.@kwdef struct Feedback{T <: Real}
    Tᶜ::T
    ΔS::T
    L::T
end

"Creates new Feedback object with updated critical temperature Tᶜ."
function updateTᶜ(Tᶜ, feedback::Feedback)
    Feedback(Tᶜ = Tᶜ, ΔS = feedback.ΔS, L = feedback.L)
end

Base.@kwdef struct Jump{T <: Real}
    j₂::T = -0.0029
    j₁::T = 0.0568
    j₀::T = -0.0577

    i₀::T = -0.25
    i₁::T = 0.95
    
    e₁::T = 2.8
    e₂::T = -0.3325
end

Base.@kwdef struct Hogg{T <: Real}
    # Defaults values
    T₀::T = 288.55 # [K]
    Tᵖ::T = 287.15 # [K]
    M₀::T = 563.88 # [p.p.m. CO₂-eq]
    Mᵖ::T = 383.15 # [p.p.m. CO₂-eq]
    N₀::T = 388.38 # [p.p.m.]

    # Climate sensitwivity
    S₀::T = 235.0 # [W m⁻²] Mean solar radiation

    ϵ::T = 1.441620917527504 # [yr J m⁻² K⁻¹] Speed of temperature
    η::T = 5.67e-8 # Stefan-Boltzmann constant 
    
    G₁::T = 22.685575408165214 # [W m⁻²] Effect of CO₂ on radiation budget
    G₀::T = 149.62958104691003 # [W m⁻²] Pre-industrial GHG radiation budget

    # Noise
    σₜ::T = 0.1441620917527504 # Standard deviation of temperature
    σₘ::T = 0.0078 # Standard deviation of CO₂e growth
    
    # Parameters of decay rate of carbon concentration
    aδ::T = 0.0176
    bδ::T = -27.63
    cδ::T = 408.3226
end

Base.broadcastable(m::Feedback) = Ref(m)
Base.broadcastable(m::Jump) = Ref(m)
Base.broadcastable(m::Hogg) = Ref(m)

"Approximation of CO₂e concentration decay."
function δₘ(M, hogg::Hogg)
    @unpack aδ, bδ, cδ, N₀, M₀ = hogg

    N = M * (N₀ / M₀)
    return aδ * exp(-(N - cδ)^2 / bδ^2)
end
function δₘ⁻¹(δ, hogg::Hogg)
    N = hogg.cδ + hogg.bδ * √(log(hogg.aδ / δ))
    return log(N * hogg.M₀ / hogg.N₀)
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

"Forcing due to greenhouse gasses."
function ghgforcing(m, hogg::Hogg)
    hogg.G₀ + hogg.G₁ * m
end

function ghgforcinginverse(r, hogg::Hogg)
    (r - hogg.G₀) / hogg.G₁
end

"Forcing due to incoming solar radiation."
@fastpow function radiativeforcing(T, hogg::Hogg)
    hogg.S₀ - hogg.η * T^4
end

"Temperature drift."
function μ(T, m, hogg::Hogg)
    ghgforcing(m, hogg) + radiativeforcing(T, hogg)
end
function μ(T, m, hogg::Hogg, feedback::Feedback)
    μ(T, m, hogg) + λ(T, feedback)
end

@fastpow function ∂μ∂T(T, m, hogg::Hogg)
    -4hogg.η * T^3 
end
function ∂μ∂T(T, m, hogg::Hogg, feedback::Feedback)
    ∂μ∂T(T, m, hogg) + λ′(T, feedback) 
end

"CO₂e log-concentration consistent with temperature T."
function mstable(T, hogg::Hogg)
    r = radiativeforcing(T, hogg)
    return ghgforcinginverse(-r, hogg)
end
function mstable(T, hogg::Hogg, feedback::Feedback)
    r = radiativeforcing(T, hogg) + λ(T, feedback)
    return ghgforcinginverse(-r, hogg)
end
function Mstable(T, hogg, args...)
    m = mstable(T, hogg, args...)

    return hogg.Mᵖ * exp(m)
end

"Temperature(s) consistent with CO₂e log-concentration m"
function Tstable(m, hogg::Hogg)
    find_zeros(T -> mstable(T, hogg) - m, 0.8hogg.Tᵖ, 2hogg.Tᵖ)
end
function Tstable(m, hogg::Hogg, feedback::Feedback)
    find_zeros(T -> mstable(T, hogg, feedback) - m, 0.8hogg.Tᵖ, 2hogg.Tᵖ)
end

const log2 = log(2)

"Compute equilibrium climate sensitivity"
function ecs(hogg::Hogg)
    only(Tstable(log2, hogg)) - hogg.Tᵖ
end
function ecs(hogg::Hogg, feedback::Feedback)
    Tstable(log2, hogg, feedback) .- hogg.Tᵖ
end

@fastpow function potential(T, m, hogg::Hogg, feedback::Feedback)
	@unpack λ₁, Δλ = feedback
    T₁ = feedback.Tᶜ + hogg.Tᵖ
    T₂ = T₁ + feedback.ΔT
    inflexion = (T₁ + T₂) / 2
	G = ghgforcing(m, hogg)

	(hogg.η / 5) * T^5 - G * T - (1 - λ₁) * hogg.S₀ * T - hogg.S₀ * Δλ * log(1 + exp(T - inflexion))
end

function density(T, m, hogg::Hogg, feedback::Feedback; normalisation = 1e-5)
    exp(-normalisation * potential(T, m, hogg, feedback))
end

"Size of jump"
function increase(T, hogg::Hogg, jump::Jump)
    ΔT = T - hogg.Tᵖ

    jump.j₀ + jump.j₁ * ΔT + jump.j₂ * ΔT^2
end


"Arrival rate of jump"
function intensity(T, hogg::Hogg, jump::Jump)
    ΔT = T - hogg.Tᵖ
    
    max(jump.i₀ + jump.i₁ / (1 + jump.e₁ * exp(jump.e₂ * ΔT)), 0)
end
