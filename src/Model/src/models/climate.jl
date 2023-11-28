const secondstoyears = 60 * 60 * 24 * 365.25
const Gtonoverppm = 1 / 7.821

Base.@kwdef struct Albedo
    Tₐ::Float64 = 3 # Transition rate
	T₁::Float64 = 290.5 # Pre-transition temperature
    T₂::Float64 = 295 # Post-transition temeprature 
     
    λ₁::Float64 = 0.31 # Pre-transition albedo
    λ₂::Float64 = 0.23 # Post-transition albedo
end

Base.@kwdef struct Hogg
    # Current and pre-industrial data temperature and carbon concentration
    T₀::Float64 = 288.56 # [K]
    Tᵖ::Float64 = 287.15 # [K]
    M₀::Float64 = 410 # [p.p.m.]
    Mᵖ::Float64 = 280 # [p.p.m.]

    N₀::Float64 = 286.65543 # [p.p.m.]
    
    σₜ::Float64 = 0.5477226 # Volatility of temperature

    # Climate sensitwivity
    S₀::Float64 = 342 # [W / m²] Mean solar radiation

    ϵ::Float64 = 5f8 / secondstoyears # years * [J / m² K] / s Heat capacity of the ocean
    η::Float64 = 5.67e-8 # Stefan-Boltzmann constant 
    
    G₁::Float64 = 20.5 # [W / m²] Effect of CO₂ on radiation budget
    G₀::Float64 = 150 # [W / m²] Pre-industrial GHG radiation budget
    
    # Decay rate of carbon concentration
    aδ::Float64 = 0.0176
    bδ::Float64 = -27.36
    cδ::Float64 = 314.8

    # Domain
    T̲::Float64 = 288.56 # Use initial levels because α > 0
    M̲::Float64 = 410 

    T̄::Float64 = 287.15 + 10 # Max. temperature, +10
    M̄::Float64 = 1010 # Concentration consistent with T̄
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
sigmoid(x) = inv(1 + exp(-x))

"Transition function"
function L(T, albedo::Albedo)
    Tᵢ = (albedo.T₁ + albedo.T₂) / 2
    return sigmoid(T - Tᵢ)
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
