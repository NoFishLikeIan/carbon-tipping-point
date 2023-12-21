using Revise
using Test: @test
using BenchmarkTools: @btime

includet("../terminalproblem.jl")

begin # Setup
    N = 101

    calibration = load_object(joinpath("data", "calibration.jld2"))
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
end;

V₀ = -1e-3 * ones(size(grid));
V = copy(V₀);
terminalpolicy = 0.5 * ones(size(grid));

terminaljacobi!(V, terminalpolicy, model);
@code_warntype terminaljacobi!(V, terminalpolicy, model);
@btime terminaljacobi!($V, $terminalpolicy, $model);