secondtoyears = 60 * 60 * 24 * 365.25
Gtonoverppm = 1 / 7.821

Base.@kwdef struct Albedo
    Tₐ::Float64 = 3. # Transition rate
	T₁::Float64 = 290.5 # Pre-transition temperature
    T₂::Float64 = 295 # Post-transition temeprature 
     
    λ₁::Float64 = 0.31 # Pre-transition albedo
    λ₂::Float64 = 0.23 # Post-transition albedo
end

Base.@kwdef struct Hogg
    # Current and pre-industrial data
    T₀::Float64 = 288.56 # [K]
    Tᵖ::Float64 = 287.15 # [K]
    M₀::Float64 = 410 # [p.p.m.]
    Mᵖ::Float64 = 280 # [p.p.m.]
    
    # Volatility
    σ²ₜ = 0.3
    σ²ₘ = 0.0

    # Climate sensitwivity
    S₀::Float64 = 342 # [W / m²] Mean solar radiation

    ϵ::Float64 = 5e8 # [J / m² K] Heat capacity of the ocean
    η::Float64 = 5.67e-8 # Stefan-Boltzmann constant 
    
    G₁::Float64 = 20.5 # [W / m²] Effect of CO₂ on radiation budget
    G₀::Float64 = 150 # [W / m²] Pre-industrial GHG radiation budget
    
    # Decay rate of carbon concentration
    aδ::Float64 = 0.0176
    bδ::Float64 = -27.36
    cδ::Float64 = 314.8
end

ClimateModel = Tuple{Hogg, Albedo}

"Decay of carbon"
function δₘ(m, baseline::Hogg)
    @unpack aδ, bδ, cδ = baseline    

    return aδ * exp(-(m - cδ)^2 / bδ^2)
end

function δₘ⁻¹(δ, baseline::Hogg)
    @unpack aδ, bδ, cδ = baseline    

    return cδ + bδ * √(log(aδ / δ))
end

# Albedo functions
"Heaviside function"
H(T, Tₐ::Real) = (1 + tanh(T / Tₐ)) / 2
H(T, albedo::Albedo) = H(T, albedo.Tₐ)


"Transition function"
function L(T, albedo::Albedo)
    @unpack T₁, T₂ = albedo
    ((T - T₁) / (T₂ - T₁)) * H(T - T₁, albedo) * H(T₂ - T, albedo) + H(T - T₂, albedo)
end

"Albedo coefficient"
λ(T, albedo::Albedo) = albedo.λ₁ - (albedo.λ₁ - albedo.λ₂) * L(T, albedo)

"Radiation dynamics"
function fₜ(T, climate::ClimateModel)
    baseline, albedo = climate
    @unpack S₀, η = baseline

    return S₀ * (1 - λ(T, albedo)) - η * T^4
end

"CO2 forcing given log of CO2 concentration"
function fₘ(m, baseline::Hogg)
    @unpack G₀, G₁, Mᵖ = baseline
    return G₀ + G₁ * (m - log(Mᵖ))
end

function fₘ⁻¹(r, baseline::Hogg)
    @unpack G₀, G₁, Mᵖ = baseline
    return log(Mᵖ) + (r - G₀) / G₁
end


"Drift temperature dynamics"
function μ(T, m, climate::ClimateModel)
    baseline = first(climate)
    return fₜ(T, climate) + fₘ(m, baseline)
end


"Compute CO₂ concentration consistent with temperature T"
function mstable(T, climate::ClimateModel)
    baseline = first(climate)
    fₘ⁻¹(-fₜ(T, climate), baseline)
end

Mstable(T, climate::ClimateModel) = exp(mstable(T, climate)) 
