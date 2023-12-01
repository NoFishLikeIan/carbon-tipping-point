using Revise
using Model

using Test: @test
using BenchmarkTools, Random
using JLD2
using FiniteDiff, Optim

rng = MersenneTwister(123)

N = 100
grid = RegularGrid([
    (Hogg().Tᵖ, Hogg().T̄), 
    (log(Hogg().M₀), log(Hogg().M̄)), 
    (log(Economy().Y̲), log(Economy().Ȳ))
], N);

@load "data/calibration.jld2" calibration;
model = ModelInstance(calibration = calibration, grid = grid, economy = Economy(ϱ = 0.));

# Mock data
t = rand(rng) * 80.;
idx = rand(rng, CartesianIndices(model.grid))
Xᵢ = model.grid.X[idx];
policy = Policy(rand(rng), Model.γ(t, model.economy, model.calibration) / 2)

# -- Benchmarking
function vfunc(X::Model.Point)
    model.economy.Y₀ * ((X.y / log(model.economy.Y₀))^2 - Model.d(X.T, model.economy, model.hogg)) - model.hogg.M₀ * (X.m / log(model.hogg.M₀))
end

V = vfunc.(model.grid.X)

Vᵢ = V[idx]
Vᵢy₊ = V[idx + Model.I[3]]
Vᵢy₋ = V[idx - Model.I[3]]
Vᵢm₊ = V[idx + Model.I[2]]

# --- Terminal Problem
Model.optimalterminalpolicy(Xᵢ, Vᵢ, Vᵢy₊, Vᵢy₋, model)
@btime Model.optimalterminalpolicy($Xᵢ, $Vᵢ, $Vᵢy₊, $Vᵢy₋, $model)

terminalpolicy = Array{Float64}(undef, size(grid));
Model.terminaljacobi!(V, terminalpolicy, model);
@btime Model.terminaljacobi!($V, $terminalpolicy, $model);