using Test, BenchmarkTools, Revise, UnPack

using Model, Grid

using Base.Threads

using SciMLBase
using ZigZagBoomerang
using Statistics
using StaticArrays
using FastClosures
using LinearAlgebra

using Optimization, OptimizationOptimJL, OptimizationPolyalgorithms
using ForwardDiff

using JLD2
using Printf, Dates

includet("../../src/extensions.jl")
includet("../utils/saving.jl")
includet("../utils/logging.jl")
includet("../markov/chain.jl")
includet("../markov/terminal.jl")
includet("../markov/backward.jl")

begin # Construct the model
	calibrationfilepath = "data/calibration.jld2"; @assert isfile(calibrationfilepath)

	calibrationfile = jldopen(calibrationfilepath, "r+")
	@unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = calibrationfile
	close(calibrationfile)
	
	damages = Kalkuhl()
	preferences = Preferences(ρ = 0.015, θ = 10., ψ = 1)
	economy = Economy(τ = calibration.τ)

	model = TippingModel(hogg, preferences, damages, economy, feedbackhigher)
	
	N = 101
	Tdomain = hogg.Tᵖ .+ (0., 5.5);
	mdomain = mstable.(Tdomain, model)
	
	G = RegularGrid((Tdomain, mdomain), N, hogg)
end;

begin # Setup terminal value function
	F̄ = ones(size(G))
	terminalconsumption = similar(F̄)
	errors = fill(Inf, size(G))

	vfi!(F̄, terminalconsumption, errors, model, G; maxiter = 10_000, verbose = 2, tol = 1e-6, alternate = true)
end;

begin # Setup problem
	ᾱ = max(γ(calibration.τ, calibration), 0)
	policy = [ Policy(terminalconsumption[idx], ᾱ) for idx in CartesianIndices(G) ]
	foc = fill(Inf, size(G))

	queue = DiagonalRedBlackQueue(G)
	Δts = zeros(prod(size(G)))
	cluster = first(ZigZagBoomerang.dequeue!(queue))

	ad = Optimization.AutoForwardDiff()
	withnegative = true
	Fₜ = copy(F̄); Fₜ₊ₕ = copy(F̄); F = (Fₜ, Fₜ₊ₕ);
end;

backwardstep!(F, policy, cluster, foc, Δts, model, calibration, G; withnegative = withnegative)
@benchmark backwardstep!($F, $policy, $cluster, $foc, $Δts, $model, $calibration, $G; withnegative = $withnegative)

Fₜ = copy(F̄); Fₜ₊ₕ = copy(F̄);
F = (Fₜ, Fₜ₊ₕ)

backwardsimulation!(F, policy, foc, model, calibration, G; verbose = 1, withnegative = withnegative, tstop = 0., tcacheinit = 10.)