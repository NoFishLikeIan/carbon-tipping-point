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

l = LinearQuadratic(τ = 0, γ = 0.15, xₛ = xₛ) # Social planner
eᵤ = (l.β₀ - l.τ) / l.β₁ # Unconstrained emissions
xₗ, xᵤ = l.xₛ, 1.5l.xₛ # Bounds on temperature


u₀ = [x₀, c₀, 0., 0.]
prob = ODEProblem(F!, u₀, (0., 2.), [m, l])

solve(prob, Tsit5(), abstol = 1e-9, reltol = 1e-9, isoutdomain = (u, t, integrator) -> any(u[1:2] .< 0.))

nullclines, equilibria = getequilibria(m, l)
ψ, ω, ϕ = nullclines 

k = 21
shadowpricespace = range(-0.5, 0.5; length = k)
emissionspace = range(-150, 150; length = k)
attractors = Array{Float64, 3}(undef, k, k, 2)

U₀ = vec(collect(Base.product(shadowpricespace, emissionspace)))

longrun = EnsembleProblem(prob;
	output_func = (sol, i) -> (sol[end], false),
	prob_func = (prob, i, repeat) -> begin
		λ₀, e₀ = U₀[i]
		@. prob.u0 = [x₀, c₀, λ₀, e₀]
		return prob
	end)

sol = solve(longrun, Tsit5(), trajectories = k^2, abstol = 1e-9, reltol = 1e-9)

