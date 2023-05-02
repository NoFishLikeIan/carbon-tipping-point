function Fexogenousemissions!(dz, z, p, t)
	m, e = p
	x, c = z

	dz[1] = m.κ * μ(x, c, m) # Temperature
	dz[2] = e(x, c) - m.δ * c # Concentration
	
	return dz
end

function F!(dz, z, p, t)
	m, l = p # Unpack LinearQuadratic and climate model
	(; κ, A, δ) = m
	(; β₀, β₁, τ, γ, ρ, xₛ) = l

	x, c, λ, e = z # Unpack state
	
	dz[1] = κ * μ(x, c, m) # Temperature 
	dz[2] = e - δ * c # Concentration 

	dz[3] = (ρ - κ * μₓ(x, m)) * λ + γ * (x - xₛ) # Shadow price of temperature
	dz[4] = (ρ + δ) * (e - (β₀ - τ) / β₁) - (λ / c) * (κ * A) / β₁ # Emissions

	return dz
end

function DF!(D, z, p, t)
	m, l = p # Unpack a LinearQuadratic model
	(; κ, A, δ) = m
	(; β₀, β₁, τ, γ, ρ, xₛ) = l

	x, c, λ, e = z # Unpack state

	J = zeros(4, 4)

	J[1, 1] = κ * μₓ(x, m)
	J[1, 2] = κ * A / c

	J[2, 2] = -δ
	J[2, 4] = 1
	
	J[3, 1] = γ - κ * μₓₓ(x, m) * λ
	J[3, 3] = ρ - κ * μₓ(x, m)

	J[4, 2] = -(κ * A / β₁) * (λ / c^2)
	J[4, 3] = -(κ * A / β₁) * (1 / c)
	J[4, 4] = ρ + δ

	D .= J

	return J
end

# Equilibrium relations assuming ċ = 0 ⟺ e = δc
"""
Get the equilibria of the deterministic climate-economy model.
"""
function getequilibria(m, l; xlims = (280, 310))
	(; κ, A, δ) = m
	(; β₀, β₁, τ, γ, ρ, xₛ) = l

	ψ(x) = φ(x, m) * δ
	ω(x) = γ * (x - xₛ) / (κ * μₓ(x, m) - ρ)
	ϕ(e) = (β₁ * e) / (κ * A * δ) * (ρ + δ) * (e - (β₀ - τ) / β₁)
	
	equilibriumcond(x) = ω(x) - (ϕ ∘ ψ)(x)
	
	asymptotesω = find_zeros(x -> κ * μₓ(x, m) - ρ, xlims[1], xlims[2])
	
	regions = [
		(xlims[1], asymptotesω[1]),
		(asymptotesω[1], asymptotesω[2]),
		(asymptotesω[2], xlims[2])
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