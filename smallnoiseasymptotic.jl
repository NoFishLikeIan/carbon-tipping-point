using EconPDEs
using DifferentialEquations
using Dierckx

using QuadGK

using Plots, StatsPlots

default(size = 600 .* (√2, 1), dpi = 220)

include("model/optimalpollution.jl")
include("utils/piecewisevalue.jl")

x̂ = 0.5 # Critical region
ξ = 0.2 # Area around steady states 

steadystates = [0., 2x̂]

N = 101

lowregime = range(-ξ, x̂ - 1e-3; length = N)
highregime = range(x̂ + 1e-3, 2x̂ + ξ; length = N)

temperature = vcat(lowregime, highregime)

"""
Returns the value function and emission function given an OptimalPollution model.
"""
function solveforvalue(m::OptimalPollution)    
    v₀ = solvepiecewisevalue(deterministic(m), (lowregime, highregime))
    v₁ = solvepiecewisevalue(firstordercorrection(m, v₀), (lowregime, highregime))
    v₂ = solvepiecewisevalue(secondordercorrection(m, v₀, v₁), (lowregime, highregime))
    
    v(x, ε; ν = 0) = v₀(x; ν) + ε * v₁(x; ν) + ε^2 * v₂(x; ν) # Value function
    e(x, ε) = E(v(x, ε; ν = 1), m) # Emissions function
    
    return v, e
end
solveforvalue(τ, γ, σ²) = solveforvalue(OptimalPollution(x̂ = x̂, τ = τ, γ = γ, σ² = σ²))

ε = 1e-3
# Monte carlo experiment
montecarlo(m::OptimalPollution) = montecarlo(solveforvalue(m)[2])
function montecarlo(e; trajectories = 1000, dt = 0.005, tspan = (0., 200.), x₀ = 0.)
    
    f(x, p, t) = -m.c * (μ(x, m.x̂) - e(x, ε))
    g(x, p, t) = ε * √σ²
    
    prob = SDEProblem(f, g, x₀, tspan)
    montecarlo = EnsembleProblem(prob)
    
    sim = solve(montecarlo, EM(); trajectories = trajectories, dt = dt)
    
    return sim
end

γ = 3.
σ² = 1.0

m = OptimalPollution(x̂ = x̂, τ = 0.01, γ = γ, σ² = σ²)
mtax = OptimalPollution(x̂ = x̂, τ = 1., γ = γ, σ² = σ²)

# Policies
v, e = solveforvalue(m)
vtax, etax = solveforvalue(mtax)

vfig = plot(xlabel = "Temperature", ylabel = "Value")

plot!(vfig, temperature, x -> v(x, ε); label = "\$\\tau = 0\$", c = :darkred)
plot!(vfig, temperature, x -> v(x, 0.); label = nothing, c = :darkred, alpha = 0.3)

plot!(vfig, temperature, x -> vtax(x, ε); c = :darkblue, label = "\$\\tau > 0\$")
plot!(vfig, temperature, x -> vtax(x, 0.); label = nothing, c = :darkblue, alpha = 0.3)

vline!(vfig, [x̂]; c = :black, label = false)
vline!(vfig, steadystates; c = :black, linestyle = :dash, label = false)

savefig(vfig, "figures/valuefunction.png")

efig = plot(xlabel = "Temperature", ylabel = "Value")

plot!(efig, temperature, x -> e(x, ε); label = "\$\\tau \\approx 0\$", c = :darkred)
plot!(efig, temperature, x -> e(x, 0.); label = nothing, c = :darkred, alpha = 0.3)

plot!(efig, temperature, x -> etax(x, ε); c = :darkblue, label = "\$\\tau > 0\$")
plot!(efig, temperature, x -> etax(x, 0.); label = nothing, c = :darkblue, alpha = 0.3)

vline!(efig, [x̂]; c = :black, label = false)
vline!(efig, steadystates; c = :black, linestyle = :dash, label = false)

savefig(efig, "figures/emissions.png")

# Monte Carlo simulation
trj = 1000
sim = montecarlo(e; trajectories = trj)
simtax = montecarlo(etax; trajectories = trj)

summ = EnsembleSummary(sim, 0:0.01:200.)
summtax = EnsembleSummary(simtax, 0:0.01:200.)

trjfig = plot(xlabel = "Time", ylabel = "Temperature", legend = :bottomright)

plot!(trjfig, summ; fillalpha = 0.2, c = :darkred, label = "\$\\tau \\approx 0\$")
plot!(trjfig, summtax; fillalpha = 0.2, c = :darkblue, label = "\$\\tau > 0\$")


hline!(trjfig, [x̂]; c = :black, label = false)
hline!(trjfig, steadystates; c = :black, linestyle = :dash, label = false)

savefig(trjfig, "figures/montecarlo.png")