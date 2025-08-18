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
	preferences = Preferences(ρ = 0.015, θ = 10., ψ = 1)
	economy = Economy()
	
	model = LinearModel(hogg, preferences, damages, economy)
	
	N = 101
	Tdomain = hogg.Tᵖ .+ (0., 10.)
	mdomain = mstable.(Tdomain, model)
	
	Gterminal = constructgrid((Tdomain, mdomain), N, hogg)
end;

begin
	state = DPState(calibration, Gterminal)
	terminaljacobi!(state, model, Gterminal)
end

vfi!(state, model, Gterminal; maxiter = 100_000, verbose = 2, tol = 1e-8, alternate = true, ω = 1.05)

if isinteractive()
	Tspace = range(G.domains[1]...; length = size(G, 1))
	mspace = range(G.domains[2]...; length = size(G, 2))
	nullcline = mstable.(Tspace, model)

	Ffig = contourf(mspace, Tspace, state.valuefunction.Fₜ; c = :viridis, xlabel = L"m", ylabel = L"T")
	plot!(Ffig, nullcline, Tspace; c = :white, linestyle = :dash, xlims = extrema(mspace), ylims = extrema(Tspace), label = false)
end