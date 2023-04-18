using DifferentialEquations, DynamicalSystems
using Roots
using LinearAlgebra


using Plots
default(size = 600 .* (√2, 1), dpi = 180, margins = 5Plots.mm, linewidth = 1.5)

include("utils/plotting.jl")
include("utils/dynamicalsystems.jl")

include("model/climate.jl")
include("model/economic.jl")


c₀ = 410 # Current carbon concentration
x₀ = first(φ⁻¹(c₀, m)) # There are three solutions to φ⁻¹(c₀), we are in the low stable temperature equilibrium

xₛ = first(φ⁻¹(m.cₚ, m)) # Surely safe temperature

# State costate dynamics
m = MendezFarazmand() # Climate model
l = LinearQuadratic(τ = 0, γ = 0.15, xₛ = xₛ) # Social planner
eᵤ = (l.β₀ - l.τ) / l.β₁ # Unconstrained emissions

xₗ, xᵤ = l.xₛ, 1.5l.xₛ # Bounds on temperature

f(x, c, λ, e) = F!(zeros(4), [x, c, λ, e], [m, l], 0.)
function F!(dz, z, p, t)
	m, l = p # Unpack a LinearQuadratic model
	x, c, λ, e = z # Unpack state
	
	dz[1] = m.κ * μ(x, c, m) # Temperature 
	dz[2] = e - m.δ * c # Concentration 

	dz[3] = (l.ρ - m.κ * ∂xμ(x, m)) * λ + l.γ * (x - l.xₛ) # Shadow price of temperature

	dz[4] = (l.ρ + m.δ) * e - m.κ * (m.A / l.β₁) * (λ / c) - eᵤ # Emission dynamics

	return dz
end

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

# Equilibrium relations assuming e = δc

"""
Get the equilibria of the deterministic climate-economy model.
"""
function getequilibria(m::MendezFarazmand, l::LinearQuadratic)
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

equilibria = getequilibria(m, l)

manifolds = computemanifolds(
	F!, DF!, equilibria, [m, l];
	dt = 0.01 	
)

# Vector field plot
λnull, enull = equilibria[1][3:4]
function g(c, x) 
	dx, dc, dλ, de = f(x, c, λnull, enull)
	
	return [dc, dx]
end
begin
	celsiustokelvin = 273.15
	narrows = 17
	npoints = 201

	tscale(n) = range(10 + celsiustokelvin, 25. + celsiustokelvin, length = n) # temperatures
	cscale(n) = range(300, 500, length = n) # concentrations

	tcnullcline = hcat(tscale(npoints), (x -> φ(x, m)).(tscale(npoints)))

	vecfig = plot(xlabel = "CO\$_2\$ concentration (ppm)", ylabel = "Temperature, K")
	plot!(vecfig, tcnullcline[:, 2], tcnullcline[:, 1], label = false, c = :black)

	plotvectorfield!(vecfig, cscale(narrows), tscale(narrows), g;rescale = 0.0001, aspect_ratio = 200 / 15)

	scatter!(vecfig, [c₀], [x₀]; label = "Current")

	for (i, z) ∈ enumerate(equilibria)
		x, c, λ, e = z
		scatter!(vecfig, [c], [x]; 
			label = i == 1 ? "Equilibrium" : false,
			c = :red
		)
	end

	vecfig
end