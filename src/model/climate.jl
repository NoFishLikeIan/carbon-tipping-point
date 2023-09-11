using UnPack

const secondtoyears = Float32(60 * 60 * 24 * 365.25)
const Gtonoverppm = Float32(1 / 7.821)

Base.@kwdef struct Albedo
    Tₐ::Float32 = 3f0 # Transition rate
	T₁::Float32 = 290.5f0 # Pre-transition temperature
    T₂::Float32 = 295f0 # Post-transition temeprature 
     
    λ₁::Float32 = 0.31f0 # Pre-transition albedo
    λ₂::Float32 = 0.23f0 # Post-transition albedo
end

Base.@kwdef struct Hogg
    # Current and pre-industrial data
    T₀::Float32 = 288.56f0 # [K]
    Tᵖ::Float32 = 287.15f0 # [K]
    M₀::Float32 = 410f0 # [p.p.m.]
    Mᵖ::Float32 = 280f0 # [p.p.m.]

    # Initial ratio of N / M
    n₀::Float32 = 286.65543f0 / 410f0

    # Domain
    T̲::Float32 = 287.15f0
    T̄::Float32 = 300f0
    m̲::Float32 = log(280f0)
    m̄::Float32 = log(1000f0)
    N̲::Float32 = 0f0
    N̅::Float32 = 314.8f0
    
    # Volatility
    σ²ₜ::Float32 = 0.3f0

    # Climate sensitwivity
    S₀::Float32 = 342f0 # [W / m²] Mean solar radiation

    ϵ::Float32 = 5f8 # [J / m² K] Heat capacity of the ocean
    η::Float32 = 5.67f-8 # Stefan-Boltzmann constant 
    
    G₁::Float32 = 20.5f0 # [W / m²] Effect of CO₂ on radiation budget
    G₀::Float32 = 150f0 # [W / m²] Pre-industrial GHG radiation budget
    
    # Decay rate of carbon concentration
    aδ::Float32 = 0.0176f0
    bδ::Float32 = -27.36f0
    cδ::Float32 = 314.8f0
end

"Decay of carbon"
function δₘ(N::Float32, hogg::Hogg)
    hogg.aδ * exp(-(N - hogg.cδ)^2 / hogg.bδ^2)
end

function δₘ⁻¹(δ::Float32, hogg::Hogg)
    hogg.cδ + hogg.bδ * √(log(hogg.aδ / δ))
end

# Albedo functions
"Heaviside function"
H(T::Float32, Tₐ::Float32) = (1 + tanh(T / Tₐ)) / 2
H(T::Float32, albedo::Albedo) = H(T, albedo.Tₐ)


"Transition function"
function L(T::Float32, albedo::Albedo)
    @unpack T₁, T₂ = albedo
    ((T - T₁) / (T₂ - T₁)) * H(T - T₁, albedo) * H(T₂ - T, albedo) + H(T - T₂, albedo)
end

"Albedo coefficient"
λ(T::Float32, albedo::Albedo) = albedo.λ₁ - (albedo.λ₁ - albedo.λ₂) * L(T, albedo)

"Radiation dynamics"
function fₜ(T::Float32, hogg::Hogg, albedo::Albedo)
    hogg.S₀ * (1 - λ(T, albedo)) - hogg.η * T^4
end

"CO2 forcing given log of CO2 concentration"
function fₘ(m::Float32, hogg::Hogg)
    hogg.G₀ + hogg.G₁ * (m - log(hogg.Mᵖ))
end

function fₘ⁻¹(r::Float32, hogg::Hogg)
    log(hogg.Mᵖ) + (r - hogg.G₀) / hogg.G₁
end


"Drift temperature dynamics"
function μ(T::Float32, m::Float32, hogg::Hogg, albedo::Albedo)
    fₜ(T, hogg, albedo) + fₘ(m, hogg)
end


"Compute CO₂ concentration consistent with temperature T"
function mstable(T::Float32, hogg::Hogg, albedo::Albedo)
    fₘ⁻¹(-fₜ(T, hogg, albedo), hogg)
end

Mstable(T::Float32, hogg::Hogg, albedo::Albedo) = exp(mstable(T, hogg, albedo)) 
