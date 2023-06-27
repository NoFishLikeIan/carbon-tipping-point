sectoyear = 3.154e7

Base.@kwdef struct MendezFarazmand
    m₀::Float64 = 410 # Current concentration
    mₚ::Float64 = 280 # Preindustrial concentration
    
    q₀::Float64 = 342 # Solar constant
    κ::Float64 = inv(5e8) * sectoyear # Scale of temperature change
    η::Float64 = 5.67e-8 # Cubic decay of temperature
    
    M::Float64 = 20.5 # Slope of concentration effect
    S::Float64 = 150 # Intercept of concentration effect
    
    δ::Float64 = 2.37e-10 * sectoyear # Concentration delay per year

    x₀::Float64 = 288.56 # Current temperature
    xₚ::Float64 = 287 # Pre-industrial temperature

    xₐ::Float64 = 3. # Transition rate
	x₁::Float64 = 289 # Pre-transition temperature
    x₂::Float64 = 296 # Post-transition temeprature 
     
    # Ice melting coefficients
    α₁::Float64 = 0.31
    α₂::Float64 = 0.2

    σ²ₓ::Float64 = 1.0
end

Climate = Union{MendezFarazmand}

H(x, climate::MendezFarazmand) = (1 + tanh(x / climate.xₐ)) / 2

Σ(x, climate::MendezFarazmand) = ((x - climate.x₁) / (climate.x₂ - climate.x₁)) * H(x - climate.x₁, climate) * H(climate.x₂ - x, climate) + H(x - climate.x₂, climate)

a(x, climate::MendezFarazmand) = climate.α₁ - (climate.α₁ - climate.α₂) * σ(x, climate)

"""Baseline temperature dynamics"""
function g(x, climate::Climate)
    @unpack q₀, α₁, α₂, η = climate
    q₀ * (1 - α₁) + q₀ * (α₁ - α₂) * Σ(x, climate) - η * x^4
end

"""Temperature dynamics with CO₂ concentration"""
function μ(x, m, climate::Climate)
    @unpack S, M, mₚ, κ = climate

    dm = S + M * log(m / mₚ) # contribution of CO₂ concentration

    return (g(x, climate) + dm) * κ
end


"""Compute CO₂ concentration consistent with temperature x"""
function nullcline(x, climate::Climate)
    @unpack S, M, mₚ = climate

    return mₚ * exp( - (g(x, climate) + S) / M ) 
end

inversenullcline(m, climate::Climate) = find_zeros(x -> nullcline(x, climate) - m, (200, 400)) # This is not a function