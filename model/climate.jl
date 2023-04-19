sectoyear = 3.154e7

Base.@kwdef struct MendezFarazmand
    cₚ::Float64 = 280 # Preindustrial concentration
    c₀::Float64 = 410 # Current concentration
    
    q₀::Float64 = 342 # Solar constant
    κ::Float64 = 5e8 / sectoyear # Scale of temperature change
    η::Float64 = 5.67e-8 # Cubic decay of temperature
    
    A::Float64 = 20.5 # Slope of concentration effect
    S::Float64 = 150 # Intercept of concentration effect
    
    δ::Float64 = 2.37e-10 * sectoyear # Concentration delay per year

    xₐ::Float64 = 3 # Transition rate
	x₁::Float64 = 289 # Pre-transition temperature
    x₂::Float64 = 295 # Post-transition temeprature 
     
    # Ice melting coefficients
    α₁::Float64 = 0.31
    α₂::Float64 = 0.2
end

σ(x, m::MendezFarazmand) = inv(1 + exp(-(x - (m.x₂ + m.x₁) / 2))) # Approximate Σ
σ′(x, m::MendezFarazmand) = σ(x, m) * (1 - σ(x, m))
σ′′(x, m::MendezFarazmand) = σ′(x, m) * (1 - 2σ(x, m))

α(x, m::MendezFarazmand) = m.α₂ * σ(x, m) + m.α₁ * (1 - σ(x, m))
α′(x, m::MendezFarazmand) = (m.α₂ - m.α₁) * σ′(x, m)
α′′(x, m::MendezFarazmand) = (m.α₂ - m.α₁) * σ′′(x, m)

function μ(x, c, m::MendezFarazmand)
    dx = m.q₀ * (1 - α(x, m)) - m.η * x^4 # natural temperature dynamics

    dc = m.S + m.A * log(c / m.cₚ) # contribution of CO₂ concentration

    return dx + dc
end

∂xμ(x, m::MendezFarazmand) = - m.q₀ * α′(x, m) - 4 * m.η * x^3
∂xxμ(x, m::MendezFarazmand) = -m.q₀ * α′′(x, m) - 12 * m.η * x^2
∂cμ(c, m::MendezFarazmand) = m.A / c

"""
Compute CO₂ concentration consistent with temperature x
"""
function φ(x, m::MendezFarazmand)
    m.cₚ * exp((m.η * x^4 - m.q₀ * (1 - α(x, m)) - m.S) / m.A) 
end

φ⁻¹(c, m::MendezFarazmand) = find_zeros(x -> φ(x, m) - c, (200, 400)) # This is not unqiue