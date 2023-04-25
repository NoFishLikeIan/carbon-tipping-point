using DifferentialEquations

using Roots
using LinearAlgebra

using Plots
default(size = 600 .* (√2, 1), dpi = 300, margins = 5Plots.mm, linewidth = 1.5)

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
γ₀ = 7.51443e-4

l = LinearQuadratic(τ = 0., γ = 1e-3, xₛ = xₛ) # Social planner
eᵤ = (l.β₀ - l.τ) / l.β₁ # Unconstrained emissions
xₗ, xᵤ = l.xₛ, 1.5l.xₛ # Bounds on temperature

nullclines, equilibria = getequilibria(m, l)
ψ, ω, ϕ = nullclines 

# Finding boundaries via shooting
reltol = 1e-10
abstol = 1e-10
isoutofdomain = (u, p, t) -> begin
	c, x, λ, e = u
	m, l = p

	stateout = c ≤ m.cₚ || x < l.xₛ || λ > 0
	emissionsout = e > eᵤ

	return stateout && emissionsout
end

timehorizons = [
	(135., 110.),
	(67., 110.),
	(30., 28.)
]

manifolds = computestablemanifolds(
	F!, DF!, equilibria, [m, l];
	alg = Rosenbrock23(), abstol = abstol, reltol = reltol,
	isoutofdomain = isoutofdomain, 
	tends = timehorizons, maxiters = 1e7,
	T = 2_000,
	h = 1e-3
)


# Plotting manifolds over (x, c), (λ, e), (x, λ), (c, e)

begin
	# -- (x, c)
	xspace = range(xₛ - 2, 299; length = 2001)
	csteadystate = (x -> φ(x, m)).(xspace)

	xticks = (280.5:5:295.5, (280:5:300) .- 273.5)
	aspect_ratio = (maximum(csteadystate) - minimum(csteadystate)) / (xspace[end] - xspace[1])

	xcfig = plot(
		xlims = extrema(csteadystate), ylims = extrema(xspace), 
		aspect_ratio = aspect_ratio,
		xlabel = "Carbon concentration \$c\$ in p.p.m.", 
		ylabel = "Temperature \$x\$ in °C",
		yticks = xticks
	)

	plot!(xcfig, csteadystate, xspace; c = :darkred, label = false)
	scatter!(xcfig, [c₀], [x₀]; c = :black, label = "Initial state")

	# -- (λ, e)
	espace = range(-25, 50; length = 1001)
	λspace = range(-650, 0; length = 1001)

	λefig = plot(
		xlims = extrema(λspace), ylims = extrema(espace), 
		aspect_ratio = (λspace[end] - λspace[1]) / (espace[end] - espace[1]),
		xlabel = "Shadow price \$\\lambda_x\$", ylabel = "Emissions \$e\$"
	)

	actionnullcine = ϕ.(espace)

	plot!(λefig, actionnullcine, espace; c = :darkred, label = false)

	# -- (x, λ)	
	xλfig = plot(
		xlims = extrema(xspace), ylims = extrema(λspace), 
		aspect_ratio = (xspace[end] - xspace[1]) / (λspace[end] - λspace[1]),
		ylabel = "Shadow price \$\\lambda_x\$", xlabel = "Temperature \$x\$",
		xticks = xticks
	)

	plot!(xλfig, xspace, ω; c = :darkred, label = nothing)
	vline!(xλfig, [x₀]; c = :black, linestyle = :dash, label = "Initial state")

	# -- (c, e)
	cspace = range(100, 1000, length = 1001)
	cefig = plot(
		ylims = extrema(espace), xlims = extrema(cspace), 
		aspect_ratio = (cspace[end] - cspace[1]) / (espace[end] - espace[1]),
		xlabel = "Carbon concentration \$c\$ in p.p.m.", 
		ylabel = "Emissions \$e\$"
	)

	plot!(cefig, cspace, c -> c * m.δ; c = :darkred, label = false)
	vline!(cefig, [c₀]; c = :black, linestyle = :dash, label = "Initial state")


	# Manifolds and steady states
	figures = [xcfig, λefig, xλfig, cefig]

	colors = [:darkgreen, :darkorange, :darkblue]

	for (i, ū) ∈ enumerate(equilibria)
		x, c, λ, e = ū
		
		stablemanifolds = manifolds[i]
		
		# -- (x, c)
		for (dir, curve) ∈ stablemanifolds
			plot!(xcfig, curve[:, 2], curve[:, 1]; c = colors[i], label = nothing)
		end
		scatter!(xcfig, [c], [x]; c = colors[i], label = nothing)

		# -- (λ, e)
		for (dir, curve) ∈ stablemanifolds
			plot!(λefig, curve[:, 3], curve[:, 4]; c = colors[i], label = nothing)
		end
		scatter!(λefig, [λ], [e]; c = colors[i], label = nothing)
		
		# -- (x, λ)
		for (dir, curve) ∈ stablemanifolds
			plot!(xλfig, curve[:, 1], curve[:, 3]; c = colors[i], label = nothing)
		end
		scatter!(xλfig, [x], [λ]; c = colors[i], label = nothing)
		
		# -- (c, e)
		for (dir, curve) ∈ stablemanifolds
			plot!(cefig, curve[:, 2], curve[:, 4]; c = colors[i], label = nothing)
		end
		scatter!(cefig, [c], [e]; c = colors[i], label = nothing)
	end

	jointfig = plot(figures..., layout = (2, 2), size = (1200, 1200))
end