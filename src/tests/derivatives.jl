using Test: @test
using BenchmarkTools
using LinearAlgebra, LoopVectorization

include("../utils/derivatives.jl")

# Tolerance for first and second derivatives
ε¹ = 1f-3
ε² = 2f-2

begin # Initialise three dimensional cube
    domains = [ (0f0, 1f0, 201), (0f0, 1f0, 201), (0f0, 1f0, 201) ];
    grid = makeregulargrid(domains);
    n = size(grid);
end;

# Generating mock data
X = Iterators.product(grid...) |> collect;

v(tup) = v(tup[1], tup[2], tup[3]);
v(T, m, y) = T^2 * y^2 + log(m + 1)

∇v(tup) = ∇v(tup[1], tup[2], tup[3]);
∇v(T, m, y) = [2T * y^2, 1 / (m + 1), 2y * T^2]

paddedslice(s) = [(1 + s):(n - s) for n ∈ size(V)] # Index without the edge
begin # Exact derivative
    V = v.(X);
    V′ = Array{Float32}(undef, size(V)..., 3);
    for idx ∈ CartesianIndices(X)
        V′[idx, :] .= ∇v(X[idx])
    end;
end

function absnorm(A, B, order)
    maximum(abs.(A - B)[paddedslice(order)..., :])
end

# Central difference
Dcentral = Array{Float32}(undef, size(V)..., length(grid));
central∇V!(Dcentral, V, grid); @time central∇V!(Dcentral, V, grid);
centralε = absnorm(Dcentral, V′, 1)
@test centralε < ε¹

# Upwind-downind difference
w = ones(Float32, size(Dcentral));
D = Array{Float32}(undef, size(V)..., length(grid) + 1);
dir∇V!(D, V, w, grid); @time dir∇V!(D, V, w, grid);

Dfwd = @view D[:, :, :, 1:3];

errors = abs.(Dfwd - V′)[paddedslice(2)..., :];
fwdε = absnorm(Dfwd, V′, 2)
@test fwdε < ε¹

# Second derivative w.r.t. the first argument
∂²Tv(T, m, y) = 2y^2
∂²TV = similar(V);
for idx ∈ CartesianIndices(V)
    ∂²TV[idx] = ∂²Tv(X[idx]...)
end

D² = similar(V);
∂²!(D², 1, V, grid); @time ∂²!(D², 1, V, grid);
T²ε = absnorm(D², ∂²TV, 2)

@test all(T²ε .< ε²)
