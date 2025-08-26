using Test, BenchmarkTools, Revise
using Plots, LaTeXStrings

using Model, Grid
using Base.Threads
using SciMLBase
using Statistics
using StaticArrays, SparseArrays
using LinearSolve, LinearAlgebra

using JLD2, UnPack
using Dates, Printf

includet("../../src/extend/model.jl")
includet("../../src/valuefunction.jl")
includet("../utils/saving.jl")
includet("../utils/logging.jl")
includet("../markov/utils.jl")
includet("../markov/chain.jl")
includet("../markov/terminal.jl")

begin # Construct the model
    calibrationfilepath = "data/calibration.jld2"
    @assert isfile(calibrationfilepath)

    calibrationfile = jldopen(calibrationfilepath, "r+")
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = calibrationfile
    close(calibrationfile)

    damages = Kalkuhl()
    preferences = Preferences()
    economy = Economy()

    threshold = Inf
    model = if 0 < threshold < Inf
        feedback = Model.updateTᶜ(threshold + hogg.Tᵖ, feedback)
        TippingModel(hogg, preferences, damages, economy, feedback)
    else
        LinearModel(hogg, preferences, damages, economy)
    end

    N = (105, 100)
    Tdomain = hogg.Tᵖ .+ (0., 10.)
    mdomain = mstable(Tdomain[1] + 0.25, model), mstable(Tdomain[2] - 0.25, model)

    G = RegularGrid(N, (Tdomain, mdomain))
end;

# Test steady state problem
valuefunction = ValueFunction(G, calibration)
Δt = 0.01
steadystate!(valuefunction, Δt, model, G, calibration; verbose = 2)

# OLD
begin
    iterations = 10_000
    indices = CartesianIndices(Gterminal)
    state = DPState(calibration.τ, Gterminal)
    # terminaljacobi!(state, model, Gterminal, indices)
    # optimalpolicy!(state, model, Gterminal)
    # vfi!(state, model, Gterminal, iterations, (indices,))
    seidelgauss!(state, model, Gterminal, iterations)
end

state, _ = computeterminal(model, Gterminal, calibration; inneriterations=10_000, verbose=1, withrichardson=true, vtol=1e-9, ptol=1e-5)

if isinteractive()
    Tspace = range(Gterminal.domains[1]...; length=size(Gterminal, 1))
    mspace = range(Gterminal.domains[2]...; length=size(Gterminal, 2))
    nullcline = mstable.(Tspace, model)

    Ffig = contourf(mspace, Tspace, state.valuefunction.Fₜ; c=:viridis, xlabel=L"m", ylabel=L"T")
    plot!(Ffig, nullcline, Tspace; c=:white, linestyle=:dash, xlims=extrema(mspace), ylims=extrema(Tspace), label=false)

    χfig = contourf(mspace, Tspace, getproperty.(state.policystate.policy, :χ); c=:Reds, xlabel=L"m", ylabel=L"T", yaxis=false)
    plot!(χfig, nullcline, Tspace; c=:white, linestyle=:dash, xlims=extrema(mspace), ylims=extrema(Tspace), label=false)

    plot(Ffig, χfig; size=600 .* (2√2, 1), margins=2Plots.mm)
end