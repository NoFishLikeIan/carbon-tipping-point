using Revise
using Model

using Test: @test
using BenchmarkTools, Random
using JLD2
using FiniteDiff, Optim
using UnPack: @unpack

rng = MersenneTwister(123)

begin # Setup
    N = 51

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

    V = load(joinpath("data", "terminal", "N=$(N)_Δλ=0.08.jld2"))["V̄"]

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

# General problem
t = economy.τ
γₜ = Model.γ(t, economy, calibration)
policy = [Policy(χ, rand() * γₜ) for χ ∈ terminalpolicy];
options = Optim.Options(allow_f_increases = true, successive_f_tol = 2);
cube = max(idx - oneunit(L), L):min(idx + oneunit(R), R);
p₀ᵢ = mean(policy[cube])

Model.optimalpolicy(t, Xᵢ, Vᵢ, Vᵢy₊, Vᵢy₋, Vᵢm₊, model; p₀ = p₀ᵢ)
@btime Model.optimalpolicy($t, $Xᵢ, $Vᵢ, $Vᵢy₊, $Vᵢy₋, Vᵢm₊, $model);

