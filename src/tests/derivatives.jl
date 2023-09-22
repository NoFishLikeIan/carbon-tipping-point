using Test: @test
using BenchmarkTools

include("../utils/derivatives.jl")

# Tolerance for first and second derivatives
ε¹ = 1f-3
ε² = 4f-2

begin # Initialise three dimensional cube
    dims = 3 # state space excluding time

    span = paddedrange(0f0, 1f0)
    idxpadsmall = 3:(length(span) - 2)
    Δ = length(span)

    X = Iterators.product(fill(span, dims)...) |> collect |> vec
    n = length(X);

    Xmat = Matrix{Float32}(undef, dims, n)
    for (i, xᵢ) ∈ enumerate(X)
        Xmat[:, i] .= xᵢ
    end
end;

# Generating mock data
v(T, m, y) = sin(T * y * m)
∇v(T, m, y) = cos(T * m * y) .* [y * m, T * y, T * m]

# Exact derivative
V = Matrix{Float32}(undef, 1, n);
V′ = Matrix{Float32}(undef, dims, n);
for i ∈ axes(V, 2)
    xᵢ = @view Xmat[:, i]
    V[i] = v(xᵢ...)
    V′[:, i] .= ∇v(xᵢ...)
end;

# Central difference
Dcentral = similar(V′);
@time central∇V!(Dcentral, V);

for dim in 1:dims
    DVmatrix = reshape(Dcentral[dim, :], Δ, Δ, Δ);
    matrix = reshape(V′[dim, :], Δ, Δ, Δ);

    errors = abs.((DVmatrix - matrix)[idxpadsmall, idxpadsmall, idxpadsmall])
    @test all(errors .< ε¹)
    println("Dimension $dim: max ε = $(maximum(errors))")
end

# Forward difference
w = -1f0ones(Float32, dims, n);
D = similar(V′);
@time dir∇V!(D, V, w);

for dim in 1:dims
    Dmatrix = reshape(D[dim, :], Δ, Δ, Δ);
    y′matrix = reshape(V′[dim, :], Δ, Δ, Δ);

    errors = abs.((Dmatrix - y′matrix)[idxpadsmall, idxpadsmall, idxpadsmall]);

    @test all(errors .< ε¹)
    println("Dimension $dim: max ε = $(maximum(errors))")
end

# Second derivative w.r.t. the first argument
∂²₁h(T, m, y) = -(m * y)^2 * sin(T * m * y)
y′′ = similar(D²);
for col in axes(y′′, 2)
    y′′[col] = ∂²₁h(X[col]...)
end

D² = similar(V);
∂²T!(D², V); @time ∂²T!(D², V);

errormatrix = reshape(abs.(y′′ - D²), Δ, Δ, Δ)[idxpadsmall, idxpadsmall, idxpadsmall];

@test all(errormatrix .< ε²)
