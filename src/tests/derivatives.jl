using Revise

using Test: @test
using BenchmarkTools
using LinearAlgebra

using Utils

# Tolerance for first and second derivatives
ε¹ = 1f-3
ε² = 2f-2

# Initialise three dimensional cube
n = 200
domains = [(1f0, 2f0, n), (0f0, 1f0, n + 1), (0f0, 1f0, n + 2)];
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

D = Array{Float32}(undef, size(grid)..., dimensions(grid) + 1);

# Boundary / Interior division
idx = CartesianIndex((1, size(grid)[2], 21));
# @code_warntype isonboundary(idx, grid);
@btime isonboundary($idx, $grid);

# Central difference
println("--- Central Difference Scheme")
@code_warntype central∇!(D, V, grid); 
@btime central∇!($D, $V, $grid);
centralε = absnorm(D[:, :, :, 1:3], V′, 1)
@test centralε < ε¹

∂₂V = similar(V);
dir = 1
@code_warntype central∂!(∂₂V, V, grid, dir);
@btime central∂!($∂₂V, $V, $grid, $dir);
@test all(∂₂V .≈ D[:, :, :, dir])

# Upwind-downind difference
println("--- Upwind scheme")
w = ones(Float32, size(grid)..., 3);
@code_warntype dir∇!(D, V, w, grid); 
@btime dir∇!($D, $V, $w, $grid);
fwdε = absnorm(D[:, :, :, 1:3], V′, 2)
@test fwdε < ε¹

dir = 2
ẏ = w[:, :, :, dir];
@code_warntype dir∂!(∂₂V, V, ẏ, grid, dir);
@btime dir∂!($∂₂V, $V, $ẏ, $grid, $dir);
@test all(∂₂V .≈ D[:, :, :, dir])

# Second derivative w.r.t. the first argument
direction = 1
∂²Tv(T, m, y) = 2y^2
∂²TV = [2y^2 for T ∈ grid.Ω[1], m ∈ grid.Ω[2], y ∈ grid.Ω[3]]

println("--- Second derivative")
D² = similar(V);
@code_warntype ∂²!(D², V, grid, direction);
@btime ∂²!($D², $V, $grid, $direction);
∂²!(D², V, grid, direction); 
T²ε = absnorm(D², ∂²TV, 2)
# @test all(T²ε .< ε²# )