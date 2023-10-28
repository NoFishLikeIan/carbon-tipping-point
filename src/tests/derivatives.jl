using Revise

using Test: @test
using BenchmarkTools
using LinearAlgebra

using OffsetArrays, ImageFiltering

using Model
using Utils

# Tolerance for first and second derivatives
ε¹ = 1f-3
ε² = 2f-2

begin # Initialise three dimensional cube
    domains = [ (0f0, 1f0, 201), (0f0, 1f0, 201), (0f0, 1f0, 201) ];
    grid = Utils.makegrid(domains);
    n = Utils.size(grid);
    dimensions = last.(domains) |> Tuple
end;

# Generating mock data
V = [T^2 * y^2 + log(m + 1) for T ∈ grid[1], m ∈ grid[2], y ∈ grid[3]];

V′ = permutedims(reinterpret(reshape, Float32, 
            [(2T * y^2, 1 / (m + 1), 2y * T^2) for T ∈ grid[1], m ∈ grid[2], y ∈ grid[3]]), 
        (2, 3, 4, 1));

function absnorm(A, B, s)
    idx = [(1 + s):(n - s) for n ∈ size(B)[1:3]]
    maximum(abs.(A - B)[idx..., :])
end

println("Testing and benchmarking:")

V = BorderArray(V, Utils.pad(V, 2));
D = Array{Float32}(undef, dimensions..., length(dimensions) + 1);

# Central difference
println("--- Central Difference Scheme")
@btime Utils.central∇!($D, $V, $grid);
centralε = absnorm(D[:, :, :, 1:3], V′, 1)
@test centralε < ε¹

∂₂V = Array{Float32}(undef, dimensions);
@btime Utils.central∂!($∂₂V, $V, $grid);
@test all(∂₂V .≈ D[:, :, :, 1])

# Upwind-downind difference
println("--- Upwind scheme")
w = ones(Float32, dimensions..., 3);
Utils.dir∇!(D, V, w, grid); 
@btime Utils.dir∇!($D, $V, $w, $grid);

fwdε = absnorm(D[:, :, :, 1:3], V′, 2)
@test fwdε < ε¹

Utils.dir∂!(∂₂V, V, w[:, :, :, 2], grid; direction = 2);
@test all(∂₂V .≈ D[:, :, :, 2])

# Second derivative w.r.t. the first argument
∂²Tv(T, m, y) = 2y^2
∂²TV = [2y^2 for T ∈ grid[1], m ∈ grid[2], y ∈ grid[3]]

println("--- Second derivative")
D² = Array{Float32}(undef, dimensions);
@btime Utils.∂²!($D², $V, $grid);
Utils.∂²!(D², V, grid); 
T²ε = absnorm(D², ∂²TV, 2);
@test all(T²ε .< ε²)