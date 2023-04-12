using EconPDEs
using DifferentialEquations
using Dierckx

using QuadGK

using Plots, StatsPlots

default(size = 600 .* (√2, 1), dpi = 220)

include("model/optimalpollution.jl")
include("utils/piecewisevalue.jl")

x̂ = 0.5 # Critical region
ξ = 0.02 # Area around steady states 

steadystates = [0., 2x̂]

N = 101

lowregime = range(-ξ, x̂ - 1e-3; length = N)
highregime = range(x̂ + 1e-3, 2x̂ + ξ; length = N)

temperature = vcat(lowregime, highregime)

"""
Returns the value function and emission function given an OptimalPollution model.
"""
function solveforvalue(m::OptimalPollution; verbose = true)
    verbose && println("Solving for v₀...")    
    v₀, resv₀ = solvepiecewisevalue(deterministic(m), (lowregime, highregime))
    verbose && println("Residual = $resv₀")    

    verbose && println("Solving for v₁...")  
    v₁, resv₁ = solvepiecewisevalue(firstordercorrection(m, v₀), (lowregime, highregime))
    verbose && println("Residual = $resv₁")    

    verbose && println("Solving for v₂...") 
    v₂, resv₂ = solvepiecewisevalue(secondordercorrection(m, v₀, v₁), (lowregime, highregime))
    verbose && println("Residual = $resv₂")

    if resv₁ > 1e-3 || resv₀ > 1e-3
        throw("Integration error too large: $resv₁, $resv₂")
    end

    if resv₂ > 1e-3
        @warn "Second order integration too large, will not use in correction."
        δ = 0
    else
        δ = 1
    end
    
    v(x, ε; ν = 0) = v₀(x; ν = ν) + ε * v₁(x; ν = ν) + δ * ε^2 * v₂(x; ν = ν) # Value function
    e(x, ε) = E(v(x, ε; ν = 1), m) # Emissions function
    
    return v, e
end

σ² = 2.
ε = 1e-4
γ = 2.0

begin # Damage
    plot(temperature, x -> d(x, -x̂, γ); label = nothing, c = :darkred, title = "Temperature damages", ylabel = "\$d_{\\gamma}(x)\$", xlabel = "Temperature")
end

# Policies
mnotax = OptimalPollution(x̂ = x̂, σ² = σ², γ = γ, τ = 0.5)
vnotax, enotax = solveforvalue(mnotax)

mtax = OptimalPollution(x̂ = x̂, σ² = σ², γ = γ, τ = 1.)
vtax, etax = solveforvalue(mtax)

begin # Value function
    vfig = plot(xlabel = "Temperature", ylabel = "Value")

    plot!(vfig, temperature, x -> vnotax(x, ε); label = "\$\\tau = 0\$", c = :darkred)
    plot!(vfig, temperature, x -> vnotax(x, 0.); label = nothing, c = :darkred, alpha = 0.3)

    plot!(vfig, temperature, x -> vtax(x, ε); c = :darkblue, label = "\$\\tau > 0\$")
    plot!(vfig, temperature, x -> vtax(x, 0.); label = nothing, c = :darkblue, alpha = 0.3)

    vline!(vfig, [x̂]; c = :black, label = false)
    vline!(vfig, steadystates; c = :black, linestyle = :dash, label = false)
end

savefig(vfig, "figures/valuefunction.png")

begin # Emissions policy
    efig = plot(xlabel = "Temperature", ylabel = "Value")

    plot!(efig, temperature, x -> enotax(x, ε); label = "\$\\tau \\approx 0\$", c = :darkred)
    plot!(efig, temperature, x -> enotax(x, 0.); label = nothing, c = :darkred, alpha = 0.3)

    plot!(efig, temperature, x -> etax(x, ε); c = :darkblue, label = "\$\\tau > 0\$")
    plot!(efig, temperature, x -> etax(x, 0.); label = nothing, c = :darkblue, alpha = 0.3)

    vline!(efig, [x̂]; c = :black, label = false)
    vline!(efig, steadystates; c = :black, linestyle = :dash, label = false)
end

savefig(efig, "figures/emissions.png")

# Monte carlo experiment
function montecarlo(m::OptimalPollution; trajectories = 1000, dt = 0.005, tspan = (0., 200.), x₀ = 0.)
    
    e = last(solveforvalue(m))
    
    f(x, p, t) = -m.c * (μ(x, m.x̂) - e(x, ε))
    g(x, p, t) = ε * √σ²
    
    prob = SDEProblem(f, g, x₀, tspan)
    montecarlo = EnsembleProblem(prob)
    
    sim = solve(montecarlo, EM(); trajectories = trajectories, dt = dt)
    
    return sim
end

begin # Monte Carlo simulation
    trj = 100
    tspan = (0., 50.)
    time = range(tspan[1], tspan[2]; length = 1000)

    sim = montecarlo(mnotax; trajectories = trj, tspan = tspan)
    simtax = montecarlo(mtax; trajectories = trj, tspan = tspan)

    summ = EnsembleSummary(sim, time)
    summtax = EnsembleSummary(simtax, time)

    trjfig = plot(xlabel = "Time", ylabel = "Temperature", legend = :bottomright)

    plot!(trjfig, summ; fillalpha = 0.2, c = :darkred, label = "\$\\tau \\approx 0\$")
    plot!(trjfig, summtax; fillalpha = 0.2, c = :darkblue, label = "\$\\tau > 0\$")


    hline!(trjfig, [x̂]; c = :black, label = false)
    hline!(trjfig, steadystates; c = :black, linestyle = :dash, label = false)
end

savefig(trjfig, "figures/montecarlo.png")