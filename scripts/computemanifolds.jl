using DrWatson
@quickactivate "scc-tipping-points" # activate the environment 

using DifferentialEquations
using Roots
using LinearAlgebra

include("../src/utils/dynamicalsystems.jl")

include("../src/model/climate.jl")
include("../src/model/economic.jl")

include("../src/statecostate/optimalpollution.jl")


# State costate dynamics

m = MendezFarazmand() # Climate model

function computemanifolds(params::Dict, timehorizons; kwargs...)
	@unpack γ, τ = params

	l = LinearQuadratic(τ = τ, γ = γ, xₛ = m.xₚ) # Social planner
	steadystates = computesteadystates(m, l)

	tipping_points = find_zeros(x -> g′(x, m), (290, 300))


	manifolds = computestablemanifolds(
		F!, DF!, steadystates, [m, l];
		alg = Rosenbrock23(), 
		tends = timehorizons, T = 2_000, h = 1e-6,
		abstol = 1e-10, reltol = 1e-10, maxiters = 1e7,
		kwargs...
	)

	results = Dict()

	for (param, value) in params
		results[param] = value
	end

	results["manifolds"] = manifolds
	results["tipping_points"] = tipping_points

	return results

end

l = LinearQuadratic()

params = Dict("γ" => 7.51443e-4, "τ" => 0.0)

tends = [
	(110., 140.),
	(110., 110.),
	(30., 28.)
]

results = computemanifolds(params, tends; verbose = true)

manifolds = results["manifolds"]

# Saving results
filename = savename(params, "jld2")
wsave(datadir("manifolds", filename), results)


tspace = range(290, 300, length = 1001)

plot(tspace, x -> ρ - κ * g′(x, m))
vline!([tipping_points[1]])
vline!([find_zero(x -> ρ - κ * g′(x, m), extrema(tspace))])

vline!([equilibria[1][1]]; label = "Stable")
vline!([equilibria[2][1]]; label = "Unstable")