using UnPack
using DotEnv, JLD2

using Lux
using NeuralPDE
using Optimization, OptimizationOptimJL
using ModelingToolkit: Interval, infimum, supremum

include("../../src/model/climate.jl")
include("../../src/model/economy.jl")

include("../../src/loss/foc.jl")
include("../../src/loss/loss.jl")

env = DotEnv.config()
DATAPATH = get(env, "DATAPATH", "data")

# Solve simplified model
economy = Economy()
@unpack A₀, δₖᵖ, κ, ϱ, θ = economy

@parameters t, y 
@variables v(..), χ(..)

Dy = Differential(y)
Dt = Differential(t)   

ϕ(t, χ) = ϕ(χ, A(t, economy), economy)
ϕ′(t, χ) = ϕ′(χ, A(t, economy), economy)
f(χ, y, u) = f(χ * exp(y), u, economy)
f′(χ, y, u) = ∂f_∂c(χ * exp(y), u, economy)

eqs = [
    f(χ(t, y), y, v(t, y)) + Dy(v(t, y)) * (ϕ(t, χ(t, y)) - δₖᵖ) + Dt(v(t, y))~ 0, # HBJ
    f′(χ(t, y), y, v(t, y)) + Dy(v(t, y)) * ϕ′(t, χ(t, y)) ~ 0 # FOC
]

T = 380.
y₀ = log(economy.Y₀)
ymin, ymax = (0.1, 10) .* y₀

bcs = [
    Dy(v(T, y)) ~ 0,
    Dt(v(T, y)) ~ -f(χ(T, y), y, v(T, y))
]

domains = [ 
    t ∈ Interval(0., T),
    y ∈ Interval(ymin, ymax) 
]

hiddensize = 80
inputsize = length(domains)
chain = [
    Lux.Chain(
        Lux.Dense(inputsize, hiddensize, Lux.relu),
        Lux.Dense(hiddensize, hiddensize, Lux.relu),
        Lux.Dense(hiddensize, hiddensize, Lux.relu),
        Lux.Dense(hiddensize, 1)
    ), # Neural network for v(t, y)
    Lux.Chain(
        Lux.Dense(inputsize, hiddensize, Lux.relu),
        Lux.Dense(hiddensize, hiddensize, Lux.relu),
        Lux.Dense(hiddensize, hiddensize, Lux.relu),
        Lux.Dense(hiddensize, 1, Lux.σ)
    ), # Neural network for χ(t, y)
]

discretization = PhysicsInformedNN(chain, QuadratureTraining())

@named pdesystem = PDESystem(eqs, bcs, domains, [t, y], [v(t, y), χ(t, y)])
prob = discretize(pdesystem, discretization)

begin # logging
    const losses = Float64[]
    sym_prob = NeuralPDE.symbolic_discretize(pdesystem, discretization)

    pdeloss = sym_prob.loss_functions.pde_loss_functions
    bcsloss = sym_prob.loss_functions.bc_loss_functions

    callback = function (p, l)
        push!(losses, l)
        print("Total loss: ", l, '\r')
        
        if length(losses) % 10 == 0
            println("HJB loss: ", map(l_ -> l_(p), pdeloss))
            println("Boundary loss: ", map(l_ -> l_(p), bcsloss))
            println("-------")
        end

        return false
    end
end

opt = OptimizationOptimJL.BFGS()
result = Optimization.solve(prob, opt; callback = callback)

println("Return code: ", result.retcode)

resultpath = joinpath(DATAPATH, "nn", "economynn.jld2")

println("Saving into ", resultpath)
save(resultpath, Dict(
    "phi" => discretization.phi,
    "depvar" => result.u.depvar,
    "domains" => domains
))