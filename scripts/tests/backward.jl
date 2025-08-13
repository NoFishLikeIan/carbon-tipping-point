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

includet("../../src/valuefunction.jl")
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
	economy = Economy()

	model = TippingModel(hogg, preferences, damages, economy, feedbackhigher)
	
	N = 31
	Tdomain = hogg.Tᵖ .+ (0., 5.5);
	mdomain = mstable.(Tdomain, model)
	
	G = RegularGrid((Tdomain, mdomain), N, hogg)
end;

begin # Setup terminal value function
	state = vfi(model, calibration, G; maxiter = 10_000, verbose = 2, tol = 1e-3, alternate = true)
end;

begin # Setup problem
	queue = DiagonalRedBlackQueue(G)
	Δts = zeros(prod(size(G)))
	cluster = first(ZigZagBoomerang.dequeue!(queue))

	ad = Optimization.AutoForwardDiff()
	withnegative = true
end;

backwardstep!(state, cluster, Δts, model, calibration, G; withnegative = withnegative)
@benchmark backwardstep!($state, $cluster, $Δts, $model, $calibration, $G; withnegative = $withnegative)

queue = DiagonalRedBlackQueue(G)
backwardsimulation!(queue, state, model, calibration, G; verbose = 1, withnegative = withnegative, tstop = 300., prevweight = 0.75, printevery = 1_000)

if isinteractive()
	using Plots
	Tspace = range(G.domains[1]...; length = size(G, 1))
	mspace = range(G.domains[2]...; length = size(G, 2))

	default(c = :viridis, label = false, dpi = 180)
	Ffig = contourf(Tspace, mspace, log.(state.valuefunction.Fₜ); title = "log(F)")

	αgrid = last.(state.policystate.policy)
	αfig = contourf(Tspace, mspace, αgrid ./ maximum(αgrid); title = "α")
	tfig = contourf(Tspace, mspace, state.timestate.t; title = "t")
	focfig = contourf(Tspace, mspace, state.policystate.foc; title = "FOC")

	plot(Ffig, αfig, tfig, focfig; layout = (2, 2), size = (800, 600))
end