secondtoyears = 60 * 60 * 24 * 365.25

Base.@kwdef struct Albedo
    xₐ::Float64 = 3. # Transition rate
	x₁::Float64 = 290.5 # Pre-transition temperature
    x₂::Float64 = 295 # Post-transition temeprature 
     
    a₁::Float64 = 0.31 # Pre-transition albedo
    a₂::Float64 = 0.23 # Post-transition albedo
end

Base.@kwdef struct Hogg
    # Current and pre-industrial data
    x₀::Float64 = 288.56 # [K]
    xₚ::Float64 = 287.15 # [K]
    m₀::Float64 = 410 # [p.p.m.]
    mₚ::Float64 = 280 # [p.p.m.]

    # Volatility
    σ²ₓ = 0.3
    σ²ₘ = 0.0

    # Climate sensitivity
    S::Float64 = 342 # [W / m²] Mean solar radiation

    ϵ::Float64 = 5e8 # [J / m² K] Heat capacity of the ocean
    η::Float64 = 5.67e-8 # Stefan-Boltzmann constant 
    
    M₁::Float64 = 20.5 # [W / m²] Effect of CO₂ on radiation budget
    M₀::Float64 = 150 # [W / m²] Pre-industrial GHG radiation budget
    
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
H(x, xₐ::Real) = (1 + tanh(x / xₐ)) / 2
H(x, albedo::Albedo) = H(x, albedo.xₐ)


"Transition function"
function L(x, albedo::Albedo)
    @unpack x₁, x₂ = albedo
    ((x - x₁) / (x₂ - x₁)) * H(x - x₁, albedo) * H(x₂ - x, albedo) + H(x - x₂, albedo)
end

"Albedo coefficient"
a(x, albedo::Albedo) = albedo.a₁ - (albedo.a₁ - albedo.a₂) * L(x, albedo)

"Radiation dynamics"
function μₓ(x, climate::ClimateModel)
    baseline, albedo = climate
    @unpack S, η = baseline

    return S * (1 - a(x, albedo)) - η * x^4
end

"CO2 forcing given log of CO2 concentration"
function μₘ(m̂, baseline::Hogg)
    @unpack M₀, M₁, mₚ = baseline
    return M₀ + M₁ * (m̂ - log(mₚ))
end

function μₘ⁻¹(r, baseline::Hogg)
    @unpack M₀, M₁, mₚ = baseline
    return log(mₚ) + (r - M₀) / M₁
end


"Drift temperature dynamics"
function μ(x, m̂, climate::ClimateModel)
    baseline = first(climate)
    return μₓ(x, climate) + μₘ(m̂, baseline)
end


"Compute CO₂ concentration consistent with temperature x"
function m̂stable(x, climate::ClimateModel)
    baseline = first(climate)
    μₘ⁻¹(-μₓ(x, climate), baseline)
end

mstable(x, climate::ClimateModel) = exp(m̂stable(x, climate)) 
