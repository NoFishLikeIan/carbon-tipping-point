using Revise

using Test: @test
using BenchmarkTools, Random
rng = MersenneTwister(123)

using FiniteDiff # To test gradients
using JLD2

using Model: hjb, objective!
using Utils

economy = Economy();
hogg = Hogg();
albedo = Albedo();

instance = (economy, hogg, albedo);
@load "data/calibration.jld2" calibration;

# -- Generate state cube
statedomain = [
    (hogg.T₀, hogg.T̄, 101), 
    (log(hogg.M₀), log(hogg.M̄), 51), 
    (log(economy.Y̲), log(economy.Ȳ), 51)
];

grid = RegularGrid(statedomain);
T = @view grid.X[:, :, :, 1];

# -- Benchmarking
V = [
    (y / log(economy.Ȳ))^2 - (exp(m) / hogg.Mᵖ)^2 *  (T / hogg.Tᵖ)^2 
    for T ∈ grid.Ω[1], m ∈ grid.Ω[2], y ∈ grid.Ω[3]
];

∇V = Array{Float32}(undef, size(grid)..., 4);
central∇!(∇V, V, grid);

∂²V = Array{Float32}(undef, size(grid));
∂²!(∂²V, V, grid, 1);

# Some sample data
begin
    t = 5f0
    idx = rand(rng, CartesianIndices(grid))
    Xᵢ = @view grid.X[idx, :]
    Vᵢ = @view V[idx]
    ∇Vᵢ = @view ∇V[idx, :]
    ∂²Vᵢ = @view ∂²V[idx]
    cᵢ = rand(Float32, 2)
end;

println("HJB given control...")
hjb(cᵢ, t, Xᵢ, Vᵢ, ∇Vᵢ, ∂²Vᵢ, instance, calibration)
@code_warntype hjb(cᵢ, t, Xᵢ, Vᵢ, ∇Vᵢ, ∂²Vᵢ, instance, calibration);
@btime hjb($cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $∂²Vᵢ, $instance, $calibration);

println("Objective functional given control...")
∇ = zeros(Float32, 2);
H = zeros(Float32, 2, 2);
Z = 0f0;
@code_warntype objective!(Z, ∇, H, cᵢ, t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration);
@btime objective!($Z, $∇, $H, $cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $instance, $calibration);

println("Optimal policy at a given point...")
@btime optimalpolicy($t, $Xᵢ, $Vᵢ, $∇Vᵢ, $instance, $calibration; c₀ = $cᵢ);

policy = ones(Float32, size(V)..., 2);
println("Computing policy over grid")
@btime policyovergrid!($policy, $t, $X, $V, $∇V, $instance, $calibration);