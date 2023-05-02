using DifferentialEquations

using Roots
using LinearAlgebra

using Plots
default(
	size = 600 .* (√2, 1), 
	dpi = 300, 
	margins = 5Plots.mm, 
	linewidth = 1.5)

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

l = LinearQuadratic(τ = 0., γ = γ₀, xₛ = xₛ) # Social planner
eᵤ = (l.β₀ - l.τ) / l.β₁ # Unconstrained emissions

nullclines, equilibria = getequilibria(m, l)
ψ, ω, ϕ = nullclines 

tipping_points = find_zeros(x -> μₓ(x, m), (290, 300))

# Finding boundaries via shooting

affect!(integrator) = terminate!(integrator, SciMLBase.ReturnCode.Success)
function instabilitycondition(u, t, integrator)
	λcond = u[3] > ω(u[1]) ? 0.0 : 1.0
	xcond = u[1] ≤ xₛ ? 0.0 : 1.0
	ccond = u[2] ≤ m.cₚ ? 0.0 : 1.0

	return λcond * xcond * ccond
end

# FIXME: Is there a better way rather than fine tuning these?
timehorizons = [
	(135., 110.),
	(67., 110.),
	(30., 28.)
]

manifolds = computestablemanifolds(
	F!, DF!, equilibria, [m, l];
	alg = Rosenbrock23(), 
	tends = timehorizons, T = 2_000, h = 1e-3,
	# callback = ContinuousCallback(instabilitycondition, affect!),
	abstol = 1e-10, reltol = 1e-10, maxiters = 1e7
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


	# -- (e, x)
	espace = range(-30, 45; length = 1001)

	xefig = plot(
		xlims = extrema(espace), ylims = extrema(xspace), 
		aspect_ratio = (espace[end] - espace[1]) / (xspace[end] - xspace[1]) ,
		ylabel = "Temperature \$x\$ in °C", xlabel = "Emissions \$e\$",
		yicks = xticks
		
	)
	
	hline!(xefig, [x₀]; c = :black, label = "Initial state", linestyle = :dash)
	

	# -- (x, λ)	
	λspace = range(-650, 0; length = 1001)
	xλfig = plot(
		xlims = extrema(xspace), ylims = extrema(λspace), 
		aspect_ratio = (xspace[end] - xspace[1]) / (λspace[end] - λspace[1]),
		ylabel = "Shadow price \$\\lambda_x\$", xlabel = "Temperature \$x\$",
		xticks = xticks
	)

	plot!(xλfig, xspace, ω; c = :darkred, label = nothing)
	vline!(xλfig, [x₀]; c = :black, linestyle = :dash, label = "Initial state")
	vline!(xλfig, tipping_points, c = :black, label = "Tipping points")

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
	figures = [xcfig, xefig, xλfig, cefig]

	colors = [:darkgreen, :darkorange, :darkblue]

	for (i, ū) ∈ enumerate(equilibria)
		x, c, λ, e = ū
		
		stablemanifolds = manifolds[i]
		
		# -- (x, c)
		for (dir, curve) ∈ stablemanifolds
			plot!(xcfig, curve[:, 2], curve[:, 1]; c = colors[i], label = nothing)
		end
		scatter!(xcfig, [c], [x]; c = colors[i], label = nothing)

		# -- (x, e)
		for (dir, curve) ∈ stablemanifolds
			plot!(xefig, curve[:, 4], curve[:, 1]; c = colors[i], label = nothing)
		end
		scatter!(xefig, [e], [x]; c = colors[i], label = nothing)
		
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
