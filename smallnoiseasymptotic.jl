using EconPDEs
using DifferentialEquations
using Dierckx

using QuadGK

using Plots, StatsPlots

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
function montecarlo(m::OptimalPollution; trajectories = 1000, dt = 0.005, tspan = (0., 200.), x₀ = 0.5)
    v, e = solveforvalue(m)
    f(x, p, t) = -m.c * (μ(x, m.x̂) - e(x, ε))
    g(x, p, t) = ε * √σ²
    
    prob = SDEProblem(f, g, x₀, tspan)
    montecarlo = EnsembleProblem(
        prob; 
        output_func = (sol, i) -> (last(sol), false) # Only return the last value
    )
    
    sim = solve(montecarlo, EM(); trajectories = trajectories, dt = dt)
    
    return sim[:]
end

γ = 4.0
σ² = 2.0


m = OptimalPollution(x̂ = x̂, τ = 0.001, γ = γ, σ² = σ²)
v, e = solveforvalue(m)

mtax = OptimalPollution(x̂ = x̂, τ = 1., γ = γ, σ² = σ²)
vt, et = solveforvalue(mtax)

# Density function
densfig = plot(xlabel = "Temperature", ylabel = "Density", legendtitle = "\$\\varepsilon\$") 
for ε ∈ [0.001, 0.006, 0.01]
    plot!(densfig, temperature, x -> φ(x, ε, e, m); label = ε)
end
densfig

# Monte Carlo simulation
trj = 100
sim = montecarlo(m; trajectories = trj)
simtax = montecarlo(mtax; trajectories = trj)

distfig = plot(xlabel = "Temperature")
histogram!(distfig, sim; label = "\$\\tau \\approx 0\$", bins = 50, normed = true, alpha = 0.5, c = :darkred)
histogram!(distfig, simtax; label = "\$\\tau = 0.6 \$", bins = 50, normed = true, alpha = 0.5, c = :darkblue)

distfig