using DifferentialEquations, DifferentialEquations.EnsembleAnalysis
using Optimization, Optim
using ForwardDiff
using StaticArrays

function drift(x, params, t)
    θ = params[1]
    return θ * x
end
function noise(x, params, t)
    _, σ, α = params
    σ * x^α
end

# True parameters
θ = 2/3
σ = 1/2
α = 2/3

x₀ = 50.
problem = SDEProblem(drift, noise, x₀, (0., 1.), (θ, σ, α))
ensembleproblem = EnsembleProblem(problem)

# Synthetic data
sol = solve(ensembleproblem; trajectories = 10_000, saveat = 0.01)
spread = timestep_quantile(sol, 0.95, :) - timestep_quantile(sol, 0.05, :)

# Optimization
function loss(p, optparams)
    ensembleproblem, spread = optparams
    σ, α = p
    θ = ensembleproblem.prob.p[1]
    
    sol = solve(ensembleproblem; p = (θ, σ, α), trajectories = 1000, saveat = 0.01)
    if !sol.converged return Inf end
    simspread = timestep_quantile(sol, 0.95, :) - timestep_quantile(sol, 0.05, :)

    return sum(abs2, simspread - spread)
end

p₀ = MVector(0.1, 0.5); loss(p₀, (ensembleproblem, spread))
objective = OptimizationFunction(loss, AutoForwardDiff())
optproblem = OptimizationProblem(objective, p₀, (ensembleproblem, spread); lb = MVector(0.01, 0.01), ub = MVector(1., 1.)) 

solve(optproblem, Fminbox(LBFGS()))