using Revise

using Test: @test
using BenchmarkTools
using LinearAlgebra

using OffsetArrays, ImageFiltering

using Utils
using Model

# Tolerance for first and second derivatives
ε¹ = 1f-3
ε² = 2f-2

# Initialise three dimensional cube
domains = [(0f0, 1f0, 201), (0f0, 1f0, 201), (0f0, 1f0, 201)];
grid = RegularGrid(domains);

# Generating mock data
V = [
    T^2 * y^2 + log(m + 1) 
    for T ∈ grid.Ω[1], m ∈ grid.Ω[2], y ∈ grid.Ω[3]
];

V′ = permutedims(reinterpret(reshape, Float32, [(2T * y^2, 1 / (m + 1), 2y * T^2) for T ∈ grid.Ω[1], m ∈ grid.Ω[2], y ∈ grid.Ω[3]]), (2, 3, 4, 1));

function absnorm(A, B, s)
    idx = [(1 + s):(n - s) for n ∈ size(B)[1:3]]
    maximum(abs.(A - B)[idx..., :])
end

println("Testing and benchmarking:")

V = BorderArray(V, paddims(V, 2));
D = Array{Float32}(undef, size(grid)..., dimensions(grid) + 1);

# Central difference
println("--- Central Difference Scheme")
@btime central∇!($D, $V, $grid);
centralε = absnorm(D[:, :, :, 1:3], V′, 1)
@test centralε < ε¹

∂₂V = similar(V.inner);
@btime central∂!($∂₂V, $V, $grid, 1);
@test all(∂₂V .≈ D[:, :, :, 1])

# Upwind-downind difference
println("--- Upwind scheme")
w = ones(Float32, size(grid)..., 3);
@btime dir∇!($D, $V, $w, $grid);

fwdε = absnorm(D[:, :, :, 1:3], V′, 2)
@test fwdε < ε¹
dir∂!(∂₂V, V, w[:, :, :, 2], grid, 2);
@test all(∂₂V .≈ D[:, :, :, 2])

# Second derivative w.r.t. the first argument
direction = 1
∂²Tv(T, m, y) = 2y^2
∂²TV = [2y^2 for T ∈ grid.Ω[1], m ∈ grid.Ω[2], y ∈ grid.Ω[3]]

println("--- Second derivative")
direction = 1
D² = similar(V.inner);
@btime ∂²!($D², $V, $grid, $direction);
∂²!(D², V, grid, direction); 
T²ε = absnorm(D², ∂²TV, 2);
@test all(T²ε .< ε²)