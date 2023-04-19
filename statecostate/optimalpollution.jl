F(z) = F!(zeros(4), z, [m, l], 0.)
function F!(dz, z, p, t)
	m, l = p # Unpack a LinearQuadratic model
	x, c, λ, e = z # Unpack state
	
	dz[1] = m.κ * μ(x, c, m) # Temperature 
	dz[2] = e - m.δ * c # Concentration 

	dz[3] = (l.ρ - m.κ * ∂xμ(x, m)) * λ + l.γ * (x - l.xₛ) # Shadow price of temperature

	dz[4] = (l.ρ + m.δ) * e - m.κ * (m.A / l.β₁) * (λ / c) - eᵤ # Emission dynamics

	return dz
end

DF(z) = DF!(zeros(4, 4), z, [m, l], 0.)
function DF!(D, z, p, t)
	m, l = p # Unpack a LinearQuadratic model
	x, c, λ, e = z # Unpack state

	J = zeros(4, 4)

	J[1, 1] = m.κ * ∂xμ(x, m)
	J[1, 2] = m.κ * m.A / c

	J[2, 2] = -m.δ
	J[2, 4] = 1
	
	J[3, 1] = l.γ - m.κ * ∂xxμ(x, m) * λ
	J[3, 3] = - m.κ * ∂xμ(x, m)

	J[4, 2] = -m.κ * (m.A / l.β₁) * (λ / c^2)
	J[4, 3] = -m.κ * (m.A / l.β₁) * (1 / c)
	J[4, 4] = l.ρ + m.δ

	D .= J

	return J
end

domain = [
	(0., Inf), # temperature
	(0., Inf), # concentration
	(-Inf, Inf), # shadow value of concentration
	(-Inf, Inf) # emissions
]

# Equilibrium relations assuming ċ = 0 ⟺ e = δc
"""
Get the equilibria of the deterministic climate-economy model.
"""
function getequilibria(m::MendezFarazmand, l::LinearQuadratic)
	eᵤ = (l.β₀ - l.τ) / l.β₁ # Unconstrained emissions

	ψ(x) = φ(x, m) * m.δ
	ω(x) = l.γ * (x - l.xₛ) / (m.κ * ∂xμ(x, m) - l.ρ)
	ϕ(e) = l.β₁ * e * ((l.ρ + m.δ) * e - eᵤ) / (m.δ * m.κ * m.A)
	
	equilibriumcond(x) = ω(x) - (ϕ ∘ ψ)(x)
	
	asymptotesω = find_zeros(x -> m.κ * ∂xμ(x, m) - l.ρ, xₗ, xᵤ)
	
	regions = [
		(xₗ, asymptotesω[1]),
		(asymptotesω[1], asymptotesω[2]),
		(asymptotesω[2], xᵤ)
	]
	
	
	equilibria = Vector{Float64}[]
	for (l, u) ∈ regions, x ∈ find_zeros(equilibriumcond, l, u)
		e = ψ(x)
		λ = ω(x)
		c = e / m.δ
	
		push!(equilibria, [x, c, λ, e])
	end
	equilibria
end