sectoyear = 3.154e7

Base.@kwdef struct MendezFarazmand
    c₀::Float64 = 410 # Current concentration
    cₚ::Float64 = 280 # Preindustrial concentration
    
    q₀::Float64 = 342 # Solar constant
    κ::Float64 = inv(5e8) * sectoyear # Scale of temperature change
    η::Float64 = 5.67e-8 # Cubic decay of temperature
    
    A::Float64 = 20.5 # Slope of concentration effect
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

H(x, m) = (1 + tanh(x / m.xₐ)) / 2
H′(x, m) = (1 / 2m.xₐ) * (1 - tanh(x / m.xₐ)^2)

Σ(x, m) = ((x - m.x₁) / (m.x₂ - m.x₁)) * H(x - m.x₁, m) * H(m.x₂ - x, m) + H(x - m.x₂, m)
Σ′(x, m) = H(x - m.x₁, m) * H(m.x₂ - x, m) / (m.x₂ - m.x₁) + ((x - m.x₁) / (m.x₂ - m.x₁)) * H′(x - m.x₁, m) * H(m.x₂ - x, m) - ((x - m.x₁) / (m.x₂ - m.x₁)) * H(x - m.x₁, m) * H′(m.x₂ - x, m) - H′(x - m.x₂, m)

σ(x) = inv(1 + exp(-x))
σ(x, m::MendezFarazmand) = σ(x - (m.x₂ + m.x₁) / 2) # Approximate Σ
σ′(x, m::MendezFarazmand) = σ(x, m) * (1 - σ(x, m))
σ′′(x, m::MendezFarazmand) = σ′(x, m) * (1 - 2σ(x, m))

α(x, m::MendezFarazmand) = m.α₁ - (m.α₁ - m.α₂) * σ(x, m)

function g(x, m::MendezFarazmand)
    (; q₀, α₁, α₂, η) = m
    q₀ * (1 - α₁) + q₀ * (α₁ - α₂) * Σ(x, m) - η * x^4
end

function μ(x, c, m::MendezFarazmand)
    (; S, A, cₚ, κ) = m

    dc = S + A * log(c / cₚ) # contribution of CO₂ concentration

    return (g(x, m) + dc) * κ
end

g′(x, m::MendezFarazmand) = m.q₀ * (m.α₁ - m.α₂) * Σ′(x, m) - 4 * m.η * x^3
g′′(x, m::MendezFarazmand) = m.q₀ * (m.α₁ - m.α₂) * σ′′(x, m) - 12 * m.η * x^2


"""
Compute CO₂ concentration consistent with temperature x
"""
function nullcline(x, m::MendezFarazmand)
    (; S, A, cₚ) = m

    return cₚ * exp( - (g(x, m) + S) / A ) 
end

inversenullcline(c, m::MendezFarazmand) = find_zeros(x -> nullcline(x, m) - c, (200, 400)) # This is not a function