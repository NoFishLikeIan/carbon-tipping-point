using Test, BenchmarkTools, Revise

using Model, Grid
using FastClosures
using ZigZagBoomerang
using Base.Threads
using SciMLBase
using Optim
using Statistics
using StaticArrays

using JLD2, UnPack
using Dates, Printf

includet("../../src/valuefunction.jl")
includet("../../src/extensions.jl")
includet("../utils/saving.jl")
includet("../utils/logging.jl")
includet("../markov/chain.jl")
includet("../markov/terminal.jl")

begin # Construct the model
	calibrationfilepath = "data/calibration.jld2"; @assert isfile(calibrationfilepath)

	calibrationfile = jldopen(calibrationfilepath, "r+")
	@unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = calibrationfile
	close(calibrationfile)
	
	damages = Kalkuhl()
	preferences = Preferences(ρ = 0.015, θ = 10., ψ = 1)
	economy = Economy()
	
	model = TippingModel(hogg, preferences, damages, economy, feedbackhigher)
	
	N = 101
	Tdomain = hogg.Tᵖ .+ (0., 5.5);
	mdomain = mstable.(Tdomain, model)
	
	G = RegularGrid((Tdomain, mdomain), N, hogg)
	state = DPState(calibration, G)
end;


terminaljacobi!(state, model, G)
vfi!(state, model, G; maxiter = 10_000, verbose = 2, tol = 1e-10, alternate = true)

if isinteractive()
	using Plots
	Tspace = range(G.domains[1]...; length = size(G, 1))
	mspace = range(G.domains[2]...; length = size(G, 2))

	contourf(mspace, Tspace, log.(state.valuefunction.Fₜ); c = :viridis)
end