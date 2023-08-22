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

# Five dimensions
xsmall = paddedrange(0.5f0, 3ϵ + 0.5f0)
idxpadsmall = 3:(length(xsmall) - 2)

# Generating data 
w₅ = reshape([1f0, 3f0, 2f0, 1f0, 1f0], 1, 5)

h(a, b, c, d, e) = exp(sin(a * b * c * d * e))
∇h(a, b, c, d, e) = cos(a * b * c * d * e) * exp(a * b * c * d * e) * [ b * c * d * e, a * c * d * e, a * b * d * e, a * b * c * e, a * b * c * d]

X = Iterators.product(fill(vec(xsmall), 5)...) |> collect |> vec;

y = reshape((z -> h.(z...)).(X), 1, length(X));
y′ = reshape((z -> first(w₅ * ∇h.(z...))).(Iterators.product(fill(vec(xsmall), 5)...)), 1, length(X));

# Testing indexing
d = size(w₅, 2)
n = size(y, 2)

Δ = floor(Int, n^(1 / d)) # Assumes n = |x|^5 

for a in eachcol(rand(1:5, 5, 5))
    @test all(X[baseidx(Δ, Tuple(a))] .- X[1] .≈ Tuple(ϵ * (a .- 1)))
end

for jdx in Iterators.product(fill(vec(1:Δ), 5)...)
    @test all(invbaseidx(Δ, baseidx(Δ, jdx)) .≈ jdx)
end

@time D = ∇₅(y, w₅);

Darray = reshape(D, Δ, Δ, Δ, Δ, Δ);
y′array = reshape(y′, Δ, Δ, Δ, Δ, Δ);

@test all(abs.((Darray - y′array)[idxpadsmall, idxpadsmall, idxpadsmall, idxpadsmall, idxpadsmall]) .< ε¹)

# Size testing
xlarge = paddedrange(0f0, 0.1f0)
d = length(xlarge)
M = reshape(rand(Float32, d, d, d, d, d), 1, d^5);


@btime ∇₅!(M, M, w₅);