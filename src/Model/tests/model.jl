using Revise
using Model

using Test: @test
using BenchmarkTools, Random
using JLD2

rng = MersenneTwister(123)

N = 100
grid = RegularGrid([
    (Hogg().Tᵖ, Hogg().T̄), 
    (log(Hogg().M₀), log(Hogg().M̄)), 
    (Economy().Y̲, Economy().Ȳ)
], N);

@load "../../data/calibration.jld2" calibration;
model = ModelInstance(calibration = calibration, grid = grid);
t = rand(rng) * 80f0;
Xᵢ = rand(rng, model.grid.X);
policy = Policy(0.5f0, Model.γ(t, model.economy, model.calibration) / 2)

# -- Benchmarking
V = [
    (Xᵢ.y / log(instance.economy.Ȳ))^2 - (exp(Xᵢ.m) / instance.hogg.Mᵖ)^2 *  (Xᵢ.T / instance.hogg.Tᵖ)^2 
    for Xᵢ ∈ grid.X
];

∇V = Array{Float64}(undef, size(grid)..., 4);
central∇!(∇V, V, grid);

∂²V = Array{Float64}(undef, size(grid));
∂²!(∂²V, V, grid, 1);

# Some sample data
begin
    t = 5f0
    idx = rand(rng, CartesianIndices(grid))
    Xᵢ = @view grid.X[idx, :]
    Vᵢ = @view V[idx]
    ∇Vᵢ = @view ∇V[idx, :]
    ∂²Vᵢ = @view ∂²V[idx]
    cᵢ = rand(Float64, 2)
end;

println("HJB given control...")
hjb(cᵢ, t, Xᵢ, Vᵢ, ∇Vᵢ, ∂²Vᵢ, instance, calibration)
@code_warntype hjb(cᵢ, t, Xᵢ, Vᵢ, ∇Vᵢ, ∂²Vᵢ, instance, calibration);
@btime hjb($cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $∂²Vᵢ, $instance, $calibration);

println("Objective functional given control...")
∇ = zeros(Float64, 2);
H = zeros(Float64, 2, 2);
Z = 0f0;
@code_warntype objective!(Z, ∇, H, cᵢ, t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration);
@btime objective!($Z, $∇, $H, $cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $instance, $calibration);