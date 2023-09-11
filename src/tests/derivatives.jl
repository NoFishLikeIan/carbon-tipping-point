using Test: @test
using BenchmarkTools

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

@time ∂(f.(x));
@time ∂²(f.(x));

@test all(abs.(∂(f.(x)) - f′.(x)[:, idxtwopad]) .< ε¹)
@test all(abs.(∂²(f.(x)) - f′′.(x)[:, idxonepad]) .< ε²)

# Two dimensions
w₂ = reshape([2f-1, 3f0], 1, 2)

g(x, y) = exp(sin(x * y))
∇g(x, y) = exp(sin(x * y)) * cos(x * y) * (w₂[1] * x + w₂[2] * y)

ĝ = g.(x', x)

@time S₂¹(w₂);
@time ∇₂(ĝ, w₂);

@test all(abs.(∇₂(ĝ, w₂) .- ∇g.(x', x)[idxtwopad, idxtwopad]) .< ε¹)

# Four dimensions
xsmall = paddedrange(0.5f0, 0.55f0)
idxpadsmall = 3:(length(xsmall) - 2)
Δ = length(xsmall)

# Generating data 
h(a, b, c, d) = exp(sin(a * b * c * d))
∇h(a, b, c, d) = cos(a * b * c * d) * exp(a * b * c * d) * [ b * c * d, a * c * d, a * b * d, a * b * c]

X = Iterators.product(fill(vec(xsmall), 4)...) |> collect |> vec;

y = reshape((z -> h.(z...)).(X), 1, length(X));

# Directional derivative
μ = ones(Float32, 1, 3)
∂αy = 2f0
∂χy = 1.5f0

V = copy(y);

@time DV = ∇V′μ(V, μ, ∂αy, ∂χy);

diry′ = similar(DV);

for col in axes(diry′, 2)
    ∇ᵢ = ∇h(X[col]...)
    diry′[1, col] = ([μ'; 1f0]'∇ᵢ)[1]
    diry′[2, col] = ([1f0; ∂αy]'∇ᵢ[[2, 3]])[1]
    diry′[3, col] = (∂χy * ∇ᵢ[[3]])[1]
end

for dim in 1:3
    DVmatrix = reshape(DV[dim, :], Δ, Δ, Δ, Δ);
    diry′matrix = reshape(diry′[dim, :], Δ, Δ, Δ, Δ);

    errors = abs.((DVmatrix - diry′matrix)[idxpadsmall, idxpadsmall, idxpadsmall, idxpadsmall]);

    @test all(errors .< ε¹)
end

# Size testing
xlarge = paddedrange(0f0, 0.25f0)

w = rand(Float32, 1, 5)
Fα = rand(Float32, 1, 2)
Fχ = rand(Float32, 1, 1)

d = length(xlarge)
M = rand(Float32, 1, d^4);

@btime ∇V′μ!(M, M, μ, ∂αy, ∂χy);