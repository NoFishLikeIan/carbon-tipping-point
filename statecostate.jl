using DifferentialEquations

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

l = LinearQuadratic(τ = 0., γ = .01, xₛ = xₛ) # Social planner
eᵤ = (l.β₀ - l.τ) / l.β₁ # Unconstrained emissions
xₗ, xᵤ = l.xₛ, 1.5l.xₛ # Bounds on temperature

nullclines, equilibria = getequilibria(m, l)
ψ, ω, ϕ = nullclines 

# Equilibria
begin
	function bif(γ)
		l = LinearQuadratic(τ = 0., γ = γ, xₛ = xₛ)
		nullclines, equilibria = getequilibria(m, l)

		return hcat(equilibria...)'[:, 1]
	end

	γspace = range(0, 2.; length = 1001)

	biffig = plot(xlabel = "\$\\gamma\$", ylabel = "Equilibrium temperature")

	for γ ∈ γspace
		equil = bif(γ)
		k = length(equil)
		scatter!(biffig, repeat([γ], k), equil; c = :black, label = false, markersize = 1)
	end

	biffig
end

# Possible paths
function isoutofdomain(u, p, t)
	c, x, λ, e = u
	stateout = c < 0 || x < 0 || λ > 0

	return stateout && (-1e3 < e < 1e3)
end

u₀ = [x₀, c₀, -100., m.δ * c₀]
prob = ODEProblem(F!, u₀, (0., 100.), [m, l])

sol = solve(prob, Tsit5(), abstol = 1e-10, reltol = 1e-10, isoutofdomain = isoutofdomain)

# Manifolds
manifolds = computestablemanifolds(
	F!, DF!, equilibria, [m, l];
	tend = 21.
)

begin
	temperaturespan = (280, 300)
	concentrationspan = (300, 600)

	tnullcline = range(temperaturespan..., length = 1001)

	fig = plot(xlims = concentrationspan, ylims = temperaturespan, xlabel = "Concentration CO\$_2\$", ylabel = "Temperature")

	plot!(fig, (x -> φ(x, m)).(tnullcline), tnullcline; c = :darkred, label = false)

	scatter!(fig, [c₀], [x₀]; c = :black, label = false)



	for (x̄, mₓ) in manifolds
		# scatter!(fig, [x̄[2]], [x̄[1]]; c = :green, label = false)
		
		for stablemanifold in mₓ
			plot!(fig, stablemanifold[2, :, 2], stablemanifold[2, :, 1]; c = :green, label = false)
			plot!(fig, stablemanifold[1, :, 2], stablemanifold[1, :, 1]; c = :green, label = false)
		end
	end

	fig
end