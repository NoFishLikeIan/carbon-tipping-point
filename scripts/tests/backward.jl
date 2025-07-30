using Test, BenchmarkTools, Revise, UnPack

using Model, Grid

using Base.Threads

using SciMLBase
using ZigZagBoomerang
using Statistics
using StaticArrays
using FastClosures
using LinearAlgebra: norm

using Optimization, OptimizationOptimJL
using ForwardDiff

using JLD2
using Printf, Dates

includet("../utils/saving.jl")
includet("../markov/chain.jl")
includet("../markov/terminal.jl")
includet("../markov/backward.jl")

begin # Construct the model
	calibrationfilepath = "data/calibration.jld2"; @assert isfile(calibrationfilepath)

	calibrationfile = jldopen(calibrationfilepath, "r+")
	@unpack hogg, calibration, albedo = calibrationfile
	close(calibrationfile)
	
	damages = GrowthDamages()
	preferences = EpsteinZin()
	economy = Economy()

	model = TippingModel(albedo, hogg, preferences, damages, economy)
end;

begin # Setup terminal value function
	N = 101
	G = terminalgrid(N, model)
	errors = Inf .* ones(size(G));
	F₀ = ones(size(G));

	F̄, terminalpolicy = vfi(F₀, model, G; maxiter = 10_000, verbose = 2, tol = 1e-5)
end;

begin # Setup problem
	policy = [MVector{2}(terminalpolicy[idx], γ(economy.τ, calibration)) for idx in CartesianIndices(G)]
	foc = [MVector{2}(Inf, Inf) for idx in CartesianIndices(G)]

	queue = DiagonalRedBlackQueue(G)
	Δts = zeros(prod(size(G)))
	cluster = first(ZigZagBoomerang.dequeue!(queue))

	ad = Optimization.AutoForwardDiff()
	withnegative = true
	Fₜ = copy(F̄); Fₜ₊ₕ = copy(F̄); F = (Fₜ, Fₜ₊ₕ);
end;

backwardstep!(Δts, F, policy, cluster, foc, model, calibration, G; withnegative = withnegative)
@benchmark backwardstep!($Δts, $F, $policy, $cluster, $foc, $model, $calibration, $G; withnegative = $withnegative)

Fₜ = copy(F̄); Fₜ₊ₕ = copy(F̄)
F = (Fₜ, Fₜ₊ₕ)

backwardsimulation!(F, policy, foc, model, calibration, G; verbose = 2, withnegative = withnegative, tstop = model.economy.τ - 0.05)