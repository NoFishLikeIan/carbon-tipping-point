using UnPack
using DotEnv, JLD2

using Lux
using NeuralPDE
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using ModelingToolkit: Interval, infimum, supremum

using Roots

include("../../src/model/climate.jl")
include("../../src/model/economy.jl")

include("../../src/loss/foc.jl")
include("../../src/loss/loss.jl")

env = DotEnv.config()
DATAPATH = get(env, "DATAPATH", "data")

# Solve simplified model
economy = Economy()
@unpack A, δₖᵖ, κ, θ = economy

@parameters y 
@variables v(..), χ(..)

Dy = Differential(y)

ϕ(χ) = ϕ(χ, economy)
ϕ′(χ) = ϕ′(χ, economy)
f(χ, y, u) = f(χ * exp(y), u, economy)
f′(χ, y, u) = ∂f_∂c(χ * exp(y), u, economy)

# Steady state
χ̄ = find_zero(χ -> ϕ(χ, economy) - δₖᵖ, (0, 1))

eqs = [
    f(χ(y), y, v(y)) + Dy(v(y)) * (ϕ(χ(y)) - δₖᵖ) ~ 0, # HJB
    f′(χ(y), y, v(y)) + Dy(v(y)) * ϕ′(χ(y)) ~ 0 # FOC
]

εbump = 1e-2
bcs = [
    bump(χ(y) - χ̄, εbump) * Dy(v(y)) ~ 0
]

y₀ = log(economy.Y₀)
ymin, ymax = (0.1, 10) .* y₀
domains = [ y ∈ Interval(ymin, ymax) ]

hiddensize = 18
inputsize = length(domains)
chain = [
    Lux.Chain(
        Lux.Dense(inputsize, hiddensize, Lux.tanh),
        Lux.Dense(hiddensize, hiddensize, Lux.tanh),
        Lux.Dense(hiddensize, hiddensize, Lux.tanh),
        Lux.Dense(hiddensize, 1)
    ), # Neural network for v(y)
    Lux.Chain(
        Lux.Dense(inputsize, hiddensize, Lux.σ),
        Lux.Dense(hiddensize, hiddensize, Lux.σ),
        Lux.Dense(hiddensize, hiddensize, Lux.σ),
        Lux.Dense(hiddensize, 1, Lux.σ)
    ), # Neural network for χ(y)
]

discretization = PhysicsInformedNN(chain, QuadratureTraining())

@named pdesystem = PDESystem(eqs, bcs, domains, [y], [v(y), χ(y)])
prob = discretize(pdesystem, discretization)

begin # logging
    losses = Float64[]
    sym_prob = NeuralPDE.symbolic_discretize(pdesystem, discretization)

    pdeloss = sym_prob.loss_functions.pde_loss_functions
    bcsloss = sym_prob.loss_functions.bc_loss_functions

    callback = function (p, l)
        push!(losses, l)
        print("Total loss: ", l, '\r')
        
        if length(losses) % 50 == 0
            println("\nHJB loss: ", map(l_ -> l_(p), pdeloss))
            println("Boundary loss: ", map(l_ -> l_(p), bcsloss))
        end

        return false
    end
end

result = Optimization.solve(
    prob, BFGS(); callback = callback, maxiters = 2000
); println("\nReturn code: ", result.retcode)

begin # Save results
    resultpath = joinpath(DATAPATH, "nn", "economynn.jld2")

    println("Saving into ", resultpath)
    save(resultpath, Dict(
        "phi" => discretization.phi,
        "depvar" => result.u.depvar,
        "domains" => domains
    ))
end

