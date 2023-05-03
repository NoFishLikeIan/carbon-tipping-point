using DrWatson
@quickactivate "scc-tipping-points" # activate the environment 

using DifferentialEquations
using Roots
using LinearAlgebra

using Interpolations

using Plots
default(size = 600 .* (√2, 1), dpi = 300, margins = 5Plots.mm, linewidth = 1.5)

include("../src/utils/dynamicalsystems.jl")

include("../src/model/climate.jl")
include("../src/model/economic.jl")

include("../src/statecostate/optimalpollution.jl")


# State costate dynamics
m = MendezFarazmand() # Climate model

timehorizons = [
	(135., 110.),
	(67., 110.),
	(30., 28.)
]

function computemanifolds(params::Dict; timehorizons = timehorizons)
	@unpack γ, τ = params

	l = LinearQuadratic(τ = τ, γ = γ, xₛ = m.xₚ) # Social planner
	nullclines, equilibria = getequilibria(m, l)

	tipping_points = find_zeros(x -> μₓ(x, m), (290, 300))

	manifolds = computestablemanifolds(
		F!, DF!, equilibria, [m, l];
		alg = Rosenbrock23(), 
		tends = timehorizons, T = 2_000, h = 1e-3,
		abstol = 1e-10, reltol = 1e-10, maxiters = 1e7
	)

	results = Dict()

	for (param, value) in params
		results[param] = value
	end

	results["manifolds"] = manifolds
	results["tipping_points"] = tipping_points

	return results

end

lowmanifolds = vcat(manifolds[1][:n], reverse(manifolds[1][:p], dims = 1))

nodes = (lowmanifolds[:, 1], lowmanifolds[:, 2])
glinear = (Gridded(Linear()), Gridded(Linear()))

λ(x, c) = interpolate(nodes, lowmanifolds[:, 3], glinear)(x, c)


params = Dict("γ" => 7.51443e-4, "τ" => 0.0)
results = computemanifolds(params)

manifolds = results["manifolds"]

# Saving results
filename = savename(params, "jld2")
wsave(datadir("manifolds", filename), results)
