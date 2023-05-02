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

    x₀::Float64 = 289 # Current temperature
    xₚ::Float64 = 287 # Pre-industrial temperature

    xₐ::Float64 = 3 # Transition rate
	x₁::Float64 = 289 # Pre-transition temperature
    x₂::Float64 = 295 # Post-transition temeprature 
     
    # Ice melting coefficients
    α₁::Float64 = 0.31
    α₂::Float64 = 0.2
end

σ(x) = inv(1 + exp(-x))
σ(x, m::MendezFarazmand) = σ(x - (m.x₂ + m.x₁) / 2) # Approximate Σ
σ′(x, m::MendezFarazmand) = σ(x, m) * (1 - σ(x, m))
σ′′(x, m::MendezFarazmand) = σ′(x, m) * (1 - 2σ(x, m))

α(x, m::MendezFarazmand) = m.α₁ - (m.α₁ - m.α₂) * σ(x, m)

function μ(x, c, m::MendezFarazmand)
    (; q₀, α₁, α₂, η, S, A, cₚ) = m

    dx = q₀ * (1 - α₁) + q₀ * (α₁ - α₂) * σ(x, m) - η * x^4 # natural temperature dynamics

    dc = S + A * log(c / cₚ) # contribution of CO₂ concentration

    return dx + dc
end

μₓ(x, m::MendezFarazmand) = m.q₀ * (m.α₁ - m.α₂) * σ′(x, m) - 4 * m.η * x^3
μₓₓ(x, m::MendezFarazmand) = m.q₀ * (m.α₁ - m.α₂) * σ′′(x, m) - 12 * m.η * x^2


"""
Compute CO₂ concentration consistent with temperature x
"""
function φ(x, m::MendezFarazmand)
    (; q₀, α₁, α₂, η, S, A, cₚ) = m

    dx = q₀ * (1 - α₁) + q₀ * (α₁ - α₂) * σ(x, m) - η * x^4 # natural temperature dynamics

    return cₚ * exp( - (dx + S) / A ) 
end

φ⁻¹(c, m::MendezFarazmand) = find_zeros(x -> φ(x, m) - c, (200, 400)) # This is not unqiue