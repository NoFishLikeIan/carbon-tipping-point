const Gtonoverppm = 1 / 7.821

struct Feedback{T <: Real}
    Tᶜ::T # Critical threshold level
    ΔT::T # Transition "width"
    β::T # Transition speed
    ΔS::T # Additional forcing
end

"Creates new Feedback object with updated critical temperature Tᶜ."
function updateTᶜ(Tᶜ, feedback::Feedback)
    Feedback(Tᶜ, feedback.ΔT, feedback.ΔS, feedback.β)
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
    T₀::T = 288.22214675250292 # [K]
    Tᵖ::T = 287.15 # [K]
    M₀::T = 558.748949198944 # [p.p.m. CO₂-eq]
    Mᵖ::T = 383.149374377142 # [p.p.m. CO₂-eq]
    N₀::T = 286.65543 # [p.p.m.]

    # Climate sensitwivity
    S₀::T = 340.5 # [W m⁻²] Mean solar radiation

    ϵ::T = 6.4 # [yr J m⁻² K⁻¹] Speed of temperature
    η::T = 5.67e-8 # Stefan-Boltzmann constant 
    
    G₁::T = 20.5 # [W m⁻²] Effect of CO₂ on radiation budget
    G₀::T = 150 # [W m⁻²] Pre-industrial GHG radiation budget

    # Noise
    σₜ::T = 1.584 # Standard deviation of temperature
    σₘ::T = 0.0078 # Standard deviation of CO₂ 
    
    # Parameters of decay rate of carbon concentration
    aδ::T = 0.0176
    bδ::T = -27.63
    cδ::T = 384.8
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
function L(T, hogg::Hogg, feedback::Feedback)
    T₁ = feedback.Tᶜ + hogg.Tᵖ
    T₂ = T₁ + feedback.ΔT
    inflexion = (T₁ + T₂) / 2

    logistic(T - inflexion, feedback.β)
end
function L′(T, hogg::Hogg, feedback::Feedback)
    T₁ = feedback.Tᶜ + hogg.Tᵖ
    T₂ = T₁ + feedback.ΔT
    inflexion = (T₁ + T₂) / 2

    sigmoid′(T - inflexion, feedback.β)
end

λ(T, hogg::Hogg, feedback::Feedback) = feedback.ΔS * L(T, hogg, feedback)
λ′(T, hogg::Hogg, feedback::Feedback) = feedback.ΔS * L′(T, hogg, feedback)

"Radiative forcing"
radiativeforcing(T, hogg::Hogg, feedback::Feedback) = hogg.S₀ * (1 - λ(T, hogg, feedback)) - hogg.η * T^4
radiativeforcing(T, hogg::Hogg) = hogg.S₀ * 0.69 - hogg.η * T^4
radiativeforcing′(T, hogg::Hogg, feedback::Feedback) = -hogg.S₀ * λ′(T, hogg, feedback) - 4hogg.η * T^3

"Greenhouse gases"
ghgforcing(m, hogg::Hogg) = hogg.G₀ + hogg.G₁ * m
ghgforcing⁻¹(r, hogg::Hogg) = (r - hogg.G₀) / hogg.G₁


"Drift temperature dynamics"
μ(T, m, hogg::Hogg, feedback::Feedback) = radiativeforcing(T, hogg, feedback) + ghgforcing(m, hogg)
μ(T, m, hogg::Hogg) = radiativeforcing(T, hogg) + ghgforcing(m, hogg)

"Compute CO₂ concentration consistent with temperature T"
mstable(T, hogg::Hogg, feedback::Feedback) = ghgforcing⁻¹(-radiativeforcing(T, hogg, feedback), hogg)
mstable(T, hogg::Hogg) = ghgforcing⁻¹(-radiativeforcing(T, hogg), hogg)

function Tstable(m, hogg::Hogg, feedback::Feedback)
    find_zeros(T -> mstable(T, hogg, feedback) - m, hogg.Tᵖ, 1.2hogg.Tᵖ)
end
function Tstable(m, hogg::Hogg)
    find_zeros(T -> mstable(T, hogg) - m, hogg.Tᵖ, 1.2hogg.Tᵖ)
end

"Compute equilibrium climate sensitivity"
function ecs(hogg::Hogg)
    Tstable(log(2), hogg) .- hogg.Tᵖ
end
function ecs(hogg::Hogg, feedback::Feedback)
    Tstable(log(2), hogg, feedback) .- hogg.Tᵖ
end

function potential(T, m, hogg::Hogg, feedback::Feedback)
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
