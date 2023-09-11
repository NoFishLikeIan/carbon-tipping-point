using UnPack

using Lux, NNlib
using ComponentArrays

using LineSearches, Optimization, OptimizationOptimJL
using LuxAMDGPU

using StatsBase

using Random
rng = Random.MersenneTwister(1234)

include("../src/model/climate.jl")
include("../src/model/economy.jl")
include("../src/model/init.jl")

include("../src/utils/derivatives.jl")
include("../src/utils/nn.jl")

function generatesample()


end

X₀ = reshape(
    [hogg.T₀, log(hogg.M₀), economy.y₀, 0f0],
    (4, 1)
) |> tounit

cube = [paddedrange(x₀, x₀ + 0.1f0) for x₀ in X₀]

const X = Matrix{Float32}(undef, 4, length(first(cube))^4)
for (i, x) in enumerate(Iterators.product(cube...))
    X[:, i] .= x
end

const n = size(X, 1)
const m = 2^6

const NN, lossfn = constructNN(n, m)

# Hot load and check allocations
Θ₀, st₀ = Lux.setup(rng, NN);
NN(X, Θ₀, st₀);
lossfn(Θ₀, st₀, X, 1f0);

const losspath = Float32[]
function callback(Θ, l, out)
	push!(losspath, l)
	return false
end

ps = ComponentArray{Float32}(Θ₀);

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((Θ, p) -> lossfn(Θ, st₀, X, 1f0), adtype)

optprob = Optimization.OptimizationProblem(optf, ps)
solver = BFGS(; initial_stepnorm = 0.01, linesearch = LineSearches.BackTracking())

# Hotload
Optimization.solve(optprob, solver; callback, maxiters = 1);
@time res = Optimization.solve(optprob, solver; callback, maxiters = 1000);