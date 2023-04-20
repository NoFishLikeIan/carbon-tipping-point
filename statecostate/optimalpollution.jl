function Fexogenousemissions!(dz, z, p, t)
	m, e = p
	x, c = z

	dz[1] = m.κ * μ(x, c, m) # Temperature
	dz[2] = e(x, c) - m.δ * c # Concentration
	
	return dz
end

F(z) = F!(zeros(4), z, [m, l], 0.)
function F!(dz, z, p, t)
	m, l = p # Unpack a LinearQuadratic model
	(; κ, A, δ) = m
	(; β₀, β₁, τ, γ, ρ, xₛ) = l

	x, c, λ, e = z # Unpack state
	
	dz[1] = κ * μ(x, c, m) # Temperature 
	dz[2] = e - δ * c # Concentration 

	dz[3] = (ρ - κ * μₓ(x, m)) * λ + γ * (x - xₛ) # Shadow price of temperature
	dz[4] = (ρ + δ) * e - (β₀ - τ) / β₁ - (κ * A * λ) / (β₁ * c) # Emissions

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

# Equilibrium relations assuming ċ = 0 ⟺ e = δc
"""
Get the equilibria of the deterministic climate-economy model.
"""
function getequilibria(m::MendezFarazmand, l::LinearQuadratic)
	(; κ, A, δ) = m
	(; β₀, β₁, τ, γ, ρ, xₛ) = l

	ψ(x) = φ(x, m) * δ
	ω(x) = γ * (x - xₛ) / (κ * μₓ(x, m) - ρ)
	ϕ(e) = e * ((ρ + δ) * β₁ * e - β₀ + τ) / (κ * A * δ)
	
	equilibriumcond(x) = ω(x) - (ϕ ∘ ψ)(x)
	
	asymptotesω = find_zeros(x -> κ * μₓ(x, m) - ρ, xₗ, xᵤ)
	
	regions = [
		(xₗ, asymptotesω[1]),
		(asymptotesω[1], asymptotesω[2]),
		(asymptotesω[2], xᵤ)
	]
	
	
	equilibria = Vector{Float64}[]
	for (l, u) ∈ regions, x ∈ find_zeros(equilibriumcond, l, u)
		e = ψ(x)
		λ = ω(x)
		c = e / δ
	
		push!(equilibria, [x, c, λ, e])
	end

	(ψ, ω, ϕ), equilibria
end