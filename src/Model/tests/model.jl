using Revise
using Model

using Test: @test
using BenchmarkTools, Random
using JLD2
using FiniteDiff, Optim

rng = MersenneTwister(123)

begin # Setup
    N = 50

    @load joinpath("data", "calibration.jld2") calibration 
    hogg = Hogg();
    economy = Economy(τ = 120.);
    albedo = Albedo();

    domains = [
        (hogg.Tᵖ, hogg.Tᵖ + 10.), 
        (log(hogg.M₀), Model.mstable(hogg.Tᵖ + 10., hogg, albedo)), 
        (log(economy.Y₀ / 2), log(economy.Y₀ * 2))
    ]

    grid = RegularGrid(domains, N);

    model = ModelInstance(
        economy = economy, hogg = hogg, albedo = albedo,
        grid = grid, calibration = calibration
    );

    V = -model.grid.h * ones(size(model.grid));

    indices = CartesianIndices(grid)
    L, R = extrema(indices)
    idx = rand(indices)

    Xᵢ = grid.X[idx]
    Vᵢ = V[idx]
    VᵢT₊, VᵢT₋ = V[min(idx + Model.I[1], R)], V[max(idx - Model.I[1], L)]
    Vᵢm₊ = V[min(idx + Model.I[2], R)]
    Vᵢy₊, Vᵢy₋ = V[min(idx + Model.I[3], R)], V[max(idx - Model.I[3], L)]
end

# Terminal problem
@btime optimalterminalpolicy($Xᵢ, $Vᵢ, $Vᵢy₊, $Vᵢy₋, $model);
terminalpolicy = Array{Float64}(undef, size(grid));
Model.terminaljacobi!(V, terminalpolicy, model);
@btime Model.terminaljacobi!($V, $terminalpolicy, $model);

# General problem
optimalpolicy(economy.τ, Xᵢ, Vᵢ, Vᵢy₊, Vᵢy₋, Vᵢm₊, model)
@btime optimalpolicy($economy.τ, $Xᵢ, $Vᵢ, $Vᵢy₊, $Vᵢy₋, Vᵢm₊, $model);
policy = [Policy(χ, 1e-3) for χ ∈ terminalpolicy]
Model.jacobi!(V, policy, economy.τ, model);
@btime Model.jacobi!($V, $policy, $economy.τ, $model);