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

@load "../../data/calibration.jld2" calibration;
model = ModelInstance(calibration = calibration, grid = grid);

# Mock data
t = rand(rng) * 80.;
idx = rand(rng, CartesianIndices(model.grid))
Xᵢ = model.grid.X[idx];
policy = Policy(rand(rng), Model.γ(t, model.economy, model.calibration) / 2)

# -- Benchmarking
vfunc(T, m, y) = model.economy.Y₀ * ((y / log(model.economy.Y₀))^2 - Model.d(T, model.economy, model.hogg)) - model.hogg.M₀ * (m / log(model.hogg.M₀));
V = [ vfunc(Xᵢ.T, Xᵢ.m, Xᵢ.y) for Xᵢ ∈ model.grid.X ];

Vᵢ = V[idx]
Vᵢy₊ = V[idx + Model.I[3]]
Vᵢy₋ = V[idx - Model.I[3]]
Vᵢm₊ = V[idx + Model.I[2]]

# Objective function
using Plots
res = Model.optimalpolicy(t, Xᵢ, Vᵢ, Vᵢy₊, Vᵢy₋, Vᵢm₊, model; policy₀ = [0.2, 0.01])


# jacobi
terminalpolicy = Array{Float64}(undef, size(grid));

V₀ = copy(V);
Model.terminaljacobi!(V, terminalpolicy, model);

function plotsection(F, m; kwargs...)
    jdx = findfirst(x -> x ≥ m, range(grid.domains[2]...; length = 100))
    ΔT, _,  Δy = grid.domains 

    Tspace = range(ΔT...; length = 100)
    yspace = range(Δy...; length = 100)

    aspect_ratio = (ΔT[2] - ΔT[1]) / (Δy[2] - Δy[1])

    contourf(Tspace, yspace, F[:, jdx, :]'; 
        aspect_ratio, xlims = ΔT, ylims = Δy, 
        xlabel = "\$T\$", ylabel = "\$y\$", kwargs...)
end