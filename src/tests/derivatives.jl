using NNlib
using SparseArrays
using Test: @test

include("../utils/derivatives.jl")

# Test
ε¹ = 1f-3
ε² = 4f-2

x = paddedrange(0f0, 1f0)
idxonepad = 2:(length(x) - 1)
idxtwopad = 3:(length(x) - 2)

# One dimension
f(x) = exp(sin(x))
f′(x) = cos(x) * exp(sin(x))
f′′(x) = exp(sin(x)) * (cos(x)^2 - sin(x)) 

@time ∂(f.(x))
@time ∂²(f.(x))

@test all(abs.(∂(f.(x)) - f′.(x)[:, idxtwopad]) .< ε¹)
@test all(abs.(∂²(f.(x)) - f′′.(x)[:, idxonepad]) .< ε²)

# Two dimensions
w₂ = reshape([0.2f0, 3f0], 1, 2)

g(x, y) = exp(sin(x * y))
∇g(x, y) = exp(sin(x * y)) * cos(x * y) * (w₂[1] * x + w₂[2] * y)

ĝ = g.(x', x)

@time ∇(ĝ, w₂)
@test all(abs.(∇(ĝ, w₂) .- ∇g.(x', x)[idxtwopad, idxtwopad]) .< ε¹)
