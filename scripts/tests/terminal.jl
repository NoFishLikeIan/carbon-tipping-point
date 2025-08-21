using Test, BenchmarkTools, Revise
using Plots, LaTeXStrings

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
	
	hogg = Hogg()
	damages = Kalkuhl()
	preferences = Preferences()
		
	threshold = 2.
	model = if 0 < threshold < Inf
		feedback = Model.updateTᶜ(threshold + hogg.Tᵖ, feedback)
		TippingModel(hogg, preferences, damages, economy, feedback)
	else
		LinearModel(hogg, preferences, damages, economy)
	end
	
	N = 101
	Tdomain = hogg.Tᵖ .+ (0., 7.)
	mdomain = mstable.(Tdomain, model)
	
	Gterminal = constructgrid((Tdomain, mdomain), N, hogg)
	Δtmax = 1 / 64
end;

begin
	state = DPState(calibration, Gterminal)
	terminaljacobi!(state, model, Gterminal)
end

vfi!(state, model, Gterminal; maxiter = 100_000, verbose = 2, tol = 1e-3, alternate = true, ω = 1.05, Δtmax = Δtmax)

if isinteractive()
	Tspace = range(Gterminal.domains[1]...; length = size(Gterminal, 1))
	mspace = range(Gterminal.domains[2]...; length = size(Gterminal, 2))
	nullcline = mstable.(Tspace, model)

	Ffig = contourf(mspace, Tspace, state.valuefunction.Fₜ; c = :viridis, xlabel = L"m", ylabel = L"T")
	plot!(Ffig, nullcline, Tspace; c = :white, linestyle = :dash, xlims = extrema(mspace), ylims = extrema(Tspace), label = false)

	χfig = contourf(mspace, Tspace, getproperty.(state.policystate.policy, :χ); c = :Reds, xlabel = L"m", ylabel = L"T", yaxis = false)
	plot!(χfig, nullcline, Tspace; c = :white, linestyle = :dash, xlims = extrema(mspace), ylims = extrema(Tspace), label = false)

	plot(Ffig, χfig; size = 300 .* (2√2, 1), margins = 3Plots.mm)
end