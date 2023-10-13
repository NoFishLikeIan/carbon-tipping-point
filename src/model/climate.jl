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
    # Current and pre-industrial data temperature and carbon concentration
    T₀::Float32 = 288.56f0 # [K]
    Tᵖ::Float32 = 287.15f0 # [K]
    M₀::Float32 = 410f0 # [p.p.m.]
    Mᵖ::Float32 = 280f0 # [p.p.m.]

    N₀::Float32 = 286.65543f0 # [p.p.m.]
    
    σ²ₜ::Float32 = 0.3f0 # Volatility of temperature

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

    # Domain
    T̲::Float32 = 288.56f0 # Use initial levels because α > 0
    M̲::Float32 = 410f0 

    T̄::Float32 = 287.15f0 + 8f0 # Max. temperature, +10
    M̄::Float32 = 645f0 # Concentration consistent with T̄
end

"Decay of carbon"
function δₘ(M, hogg::Hogg)
    N = M * (hogg.N₀ / hogg.M₀)
    return hogg.aδ * exp(-(N - hogg.cδ)^2 / hogg.bδ^2)
end

function δₘ⁻¹(δ, hogg::Hogg)
    N = hogg.cδ + hogg.bδ * √(log(hogg.aδ / δ))
    return log(N * hogg.M₀ / hogg.N₀)
end

# Albedo functions
"Heaviside function"
H(T, Tₐ) = (1 + tanh(T / Tₐ)) / 2
H(T, albedo::Albedo) = H(T, albedo.Tₐ)


"Transition function"
function L(T, albedo::Albedo)
    @unpack T₁, T₂ = albedo
    ((T - T₁) / (T₂ - T₁)) * H(T - T₁, albedo) * H(T₂ - T, albedo) + H(T - T₂, albedo)
end

"Albedo coefficient"
λ(T, albedo::Albedo) = albedo.λ₁ - (albedo.λ₁ - albedo.λ₂) * L(T, albedo)

"Radiation dynamics"
function fₜ(T, hogg::Hogg, albedo::Albedo)
    hogg.S₀ * (1 - λ(T, albedo)) - hogg.η * T^4
end

"CO2 forcing given log of CO2 concentration"
function fₘ(m, hogg::Hogg)
    hogg.G₀ + hogg.G₁ * (m - log(hogg.Mᵖ))
end

function fₘ⁻¹(r, hogg::Hogg)
    log(hogg.Mᵖ) + (r - hogg.G₀) / hogg.G₁
end


"Drift temperature dynamics"
function μ(T, m, hogg::Hogg, albedo::Albedo)
    fₜ(T, hogg, albedo) + fₘ(m, hogg)
end


"Compute CO₂ concentration consistent with temperature T"
function mstable(T, hogg::Hogg, albedo::Albedo)
    fₘ⁻¹(-fₜ(T, hogg, albedo), hogg)
end

Mstable(T, hogg::Hogg, albedo::Albedo) = exp(mstable(T, hogg, albedo)) 
