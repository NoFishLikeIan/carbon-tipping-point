using Revise
using Model

using Test: @test
using BenchmarkTools, Random
using JLD2
using UnPack: @unpack

rng = MersenneTwister(123)

begin # Setup
    N = 51

    datapath = "../../data"

    calibration = load_object(joinpath(datapath, "calibration.jld2"))
    hogg = Hogg();
    economy = Economy();
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

    V = load(joinpath(datapath, "terminal", "N=$(N)_Δλ=0.08.jld2"))["V̄"]
    
    policy = [Policy(χ, 1e-5) for χ ∈ load(joinpath(datapath, "terminal", "N=$(N)_Δλ=0.08.jld2"))["policy"]]

    indices = CartesianIndices(grid)
end;

# Interpolations
xs = [xᵢ + rand(rng, 3) / 10 for xᵢ ∈ grid.X[idx:(idx + oneunit(idx))]];
@btime interpolateovergrid($grid, $V, $xs);

# Grid
