function hₑ(pc, l::LinearQuadratic)
    (; ē, β₁, β₀, τ) = l

    if pc ≤ τ - β₀
        return 0.
    elseif pc ≥ β₁ * ē + τ - β₀
        return ē
    else
        return (β₀ - τ + pc) / β₁
    end
end

function hₑ′(pc, l::LinearQuadratic)
	(; ē, β₁, β₀, τ) = l

	if pc ≥ β₁ * ē + τ - β₀ || pc ≤ τ - β₀
		return 0.
	end

	return 1 / β₁
end

function F!(dz, z, p, t)
	m, l = p # Unpack LinearQuadratic and climate model
	(; κ, A, δ) = m
	(; γ, ρ, xₛ) = l

	x, c, λx, λc = z # Unpack state
	
	dz[1] = μ(x, c, m) # Temperature 
	dz[2] = hₑ(λc, l) - δ * c # Concentration 

	dz[3] = (ρ - κ * g′(x, m)) * λx + γ * (x - xₛ) # Shadow price of temperature
	dz[4] = (ρ + δ) * λc - (κ * A) * λx / c # Emissions

	return dz
end

function DF!(D, z, p, t)
	m, l = p # Unpack LinearQuadratic and climate model
	(; κ, A, δ) = m
	(; γ, ρ) = l

	x, c, λx, λc = z # Unpack state

	J = zeros(4, 4)

	J[1, 1] = κ * g′(x, m)
	J[1, 2] = κ * A / c

	J[2, 2] = -δ
	J[2, 4] = hₑ′(λc, l)
	
	J[3, 1] = γ - κ * g′′(x, m) * λx
	J[3, 3] = ρ - κ * g′(x, m)

	J[4, 2] = (κ * A) * (λx / c^2)
	J[4, 3] = -(κ * A) / c
	J[4, 4] = ρ + δ

	D .= J

	return J
end

function H(x, c, px, pc, m::MendezFarazmand, l::LinearQuadratic)
	e = h(pc, l)
	u(e, l) - d(x, l) + px * μ(x, c, m) + pc * (e - δ * c)
end

# Equilibrium relations assuming ċ = 0 ⟺ e = δc
"""
Get the equilibria of the deterministic climate-economy model.
"""
function computesteadystates(m, l; xlimits = (280, 310), ε = 1e-9)
	(; κ, A, δ) = m
	(; γ, ρ, xₛ) = l

	function o(x)
		c = φ(x, m)
		λx = γ * (x - xₛ) / (κ * g′(x, m) - ρ)
		λc = (κ * A * λx) / (c * (ρ + δ))

		return hₑ(λc, l) - δ * c
	end

	asymptotes = find_zeros(x -> κ * g′(x, m) - ρ, xlimits...)	
	tipping_points = find_zeros(x -> g′(x, m), xlimits...)
	regions = [
		(xlimits[1], asymptotes[1]),
		(tipping_points[1], asymptotes[1]),
		(asymptotes[2], tipping_points[2]),
		(asymptotes[2], xlimits[2])
	]
	
	
	equilibria = Vector{Float64}[]
	for (xl, xu) ∈ regions, x ∈ find_zero(o, (xl, xu))
		c = φ(x, m)
		λx = γ * (x - xₛ) / (κ * g′(x, m) - ρ)
		λc = (κ * A * λx) / (c * (ρ + δ))	
		push!(equilibria, [x, c, λx, λc])
	end

	return equilibria
end