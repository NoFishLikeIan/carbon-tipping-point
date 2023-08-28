using Test: @test
using BenchmarkTools
using Lux

using Random
rng = Random.MersenneTwister(1234);

include("../model/init.jl")
include("../utils/derivatives.jl")
include("../utils/nn.jl")

unit = paddedrange(0f0, 0.025f0)
cube = fill(unit, 5)

X = Matrix{Float32}(undef, 5, length(unit)^5)
for (i, x) in enumerate(Iterators.product(cube...))
    X[:, i] .= x
end
@test all(tounit(fromunit(X)) .≈ X)

α = rand(Float32, 1, size(X, 2))
χ = rand(Float32, 1, size(X, 2))
V = rand(Float32, 1, size(X, 2))

# Time utility functions
@time Eᵇ.(X[[5], :])
@time Fα(X, α)
@time Fχ(X, χ)

# Test chains
n = size(X, 1)
m = 2^7

NN, lossfn = constructNN(n, m)
Θ, st = Lux.setup(rng, NN)

@time NN(X, Θ, st)
@time lossfn(Θ, st, X, 1f0)