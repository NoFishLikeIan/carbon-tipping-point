using Revise
using Test: @test
using BenchmarkTools: @btime

includet("../terminalproblem.jl")

begin # Setup
    N = 50

    calibration = load_object(joinpath("data", "calibration.jld2"))
    hogg = Hogg();
    economy = Economy();
    albedo = Albedo();

    domains = [
        (hogg.Tᵖ, hogg.Tᵖ + 10.), 
        (log(hogg.M₀), Model.mstable(hogg.Tᵖ + 10., hogg, albedo)), 
        (log(economy.Y₀ / 2), log(economy.Y₀ * 2))
    ]

    grid = RegularGrid(domains, N);

    model = ModelInstance(economy, hogg, albedo, grid, calibration);
end;

V₀ = -ones(size(grid));
V = copy(V₀);
terminalpolicy = 0.5 * ones(size(grid));

terminaljacobi!(V, terminalpolicy, model);
@code_warntype terminaljacobi!(V, terminalpolicy, model);
@btime terminaljacobi!($V, $terminalpolicy, $model);