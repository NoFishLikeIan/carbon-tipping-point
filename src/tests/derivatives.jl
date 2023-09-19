using Test: @test
using BenchmarkTools

include("../utils/derivatives.jl")

# Test
ε¹ = 1f-3
ε² = 4f-2

# Four dimensions
xsmall = paddedrange(0.5f0, 0.7f0)
idxpadsmall = 3:(length(xsmall) - 2)
Δ = length(xsmall)

# Generating data 
h(a, b, c, d) = exp(sin(a * b * c * d))
∇h(a, b, c, d) = cos(a * b * c * d) * exp(a * b * c * d) * [ b * c * d, a * c * d, a * b * d, a * b * c]

X = Iterators.product(fill(vec(xsmall), 4)...) |> collect |> vec;

V = reshape((z -> h.(z...)).(X), 1, length(X));

# Directional derivative
w = ones(Float32, 3, size(V, 2));

DV = similar(V, 4, size(V, 2));
@time dir∇V!(DV, V, w);

diry′ = similar(DV);
for col in axes(diry′, 2)
    ∇ᵢ = ∇h(X[col]...)
    diry′[1, col] = ∇ᵢ[1]
    diry′[2, col] = ∇ᵢ[3]
    diry′[3, col] = ∇ᵢ[4]
    diry′[4, col] = sum(∇ᵢ[2:4])
end

for dim in 1:3
    DVmatrix = reshape(DV[dim, :], Δ, Δ, Δ, Δ);
    diry′matrix = reshape(diry′[dim, :], Δ, Δ, Δ, Δ);

    errors = abs.((DVmatrix - diry′matrix)[idxpadsmall, idxpadsmall, idxpadsmall, idxpadsmall]);

    @test all(errors .< ε¹)
end

# Second derivative w.r.t. the second argument
∂²₂h(a, b, c, d) = (a * c * d)^2 * (cos(a * b * c * d)^2 - sin(a * b * c * d)^2) * h(a, b, c, d)

D² = similar(V);
∂²T!(D², V); @time ∂²T!(D², V);

y′′ = similar(D²);
for col in axes(y′′, 2)
    y′′[col] = ∂²₂h(X[col]...)
end

errormatrix = reshape(abs.(y′′ - D²), Δ, Δ, Δ, Δ)[idxpadsmall, idxpadsmall, idxpadsmall, idxpadsmall];

@test all(errormatrix .< ε²)
