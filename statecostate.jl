using DifferentialEquations
using DynamicalSystems

using Roots
using LinearAlgebra


using Plots
default(size = 600 .* (√2, 1), dpi = 180, margins = 5Plots.mm, linewidth = 1.5)

include("utils/plotting.jl")
include("utils/dynamicalsystems.jl")

include("model/climate.jl")
include("model/economic.jl")

include("statecostate/optimalpollution.jl")


# State costate dynamics
m = MendezFarazmand() # Climate model
c₀ = 410 # Current carbon concentration
x₀ = first(φ⁻¹(c₀, m)) # Current temperature
xₛ = first(φ⁻¹(m.cₚ, m)) # Surely safe temperature
eᵤ = (l.β₀ - l.τ) / l.β₁ # Unconstrained emissions

l = LinearQuadratic(τ = 0, γ = 100., xₛ = xₛ) # Social planner
xₗ, xᵤ = l.xₛ, 1.5l.xₛ # Bounds on temperature

equilibria = getequilibria(m, l)

solver = (alg = TanYam7(), abstol = 1.0e-8, reltol = 1.0e-8, dt = 1e-2)

# Vector field plot
λnull, enull = equilibria[1][3:4]
function g(c, x) 
	dx, dc, dλ, de = F([x, c, λnull, enull])
	
	return [dc, dx]
end
	
u₀ = [x₀, c₀, λnull, enull]
ds = TangentDynamicalSystem(
	CoupledODEs(F!, u₀, [m, l]; solver...),
	J = DF!, J0 = DF(u₀)
)

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
		
		J = DF!(zeros(4, 4), z, [m, l], 0.)
		λ, v = eigen(J)

		scatter!(vecfig, [c], [x]; 
			label = i == 1 ? "Equilibrium" : false,
			c = :red
		)
	end

	vecfig
end