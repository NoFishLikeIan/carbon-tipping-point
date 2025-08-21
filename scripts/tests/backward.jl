using Test, BenchmarkTools, Revise, UnPack
using Plots, LaTeXStrings
default(c=:viridis, label=false, dpi=180)

using Model, Grid

using Base.Threads

using SciMLBase
using ZigZagBoomerang
using Statistics
using StaticArrays
using FastClosures
using LinearAlgebra

using Optimization, SimpleOptimization
using OptimizationOptimJL, OptimizationPolyalgorithms, LineSearches

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
    calibrationfilepath = "data/calibration.jld2"
    @assert isfile(calibrationfilepath)

    calibrationfile = jldopen(calibrationfilepath, "r+")
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = calibrationfile
    close(calibrationfile)

    hogg = Hogg()
    damages = Kalkuhl()
    preferences = Preferences()
    economy = Economy()
    threshold = 2.0

    model = if 0 < threshold < Inf
        feedback = Model.updateTᶜ(threshold + hogg.Tᵖ, feedback)
        TippingModel(hogg, preferences, damages, economy, feedback)
    else
        LinearModel(hogg, preferences, damages, economy)
    end

    # Construct Grid
    N = 101
    Tdomain = hogg.Tᵖ .+ (0., 10.)
    mdomain = mstable.(Tdomain, model)
    Gterminal = constructgrid((Tdomain, mdomain), N, hogg)
    terminalstate = DPState(calibration, Gterminal)

    Δtmax = 1 / 100
end;

vfi!(terminalstate, model, Gterminal; maxiter=100_000, tol=1e-8, alternate=true, ω=1., verbose=2, Δtmax=Δtmax)

if isinteractive()
    Tspace = range(Gterminal.domains[1]...; length=size(Gterminal, 1))
    mspace = range(Gterminal.domains[2]...; length=size(Gterminal, 2))

    Ffig = contourf(mspace, Tspace, terminalstate.valuefunction.Fₜ; title=L"\log(F)", ylabel=L"T", xlabel=L"m", margins=5Plots.mm)
end

begin # Setup problem
    G = shrink(Gterminal, 0.9, hogg)
    state = interpolateovergrid(terminalstate, Gterminal, G)
    queue = DiagonalRedBlackQueue(G)
    Δts = zeros(prod(size(G)))
    cluster = first(ZigZagBoomerang.dequeue!(queue))

    ad = Optimization.AutoForwardDiff()
    withnegative = false
    T = Float64
    alg = LBFGS() # BFGS(initial_invH = inverseidentity)

    @unpack valuefunction, policystate, timestate = state
end;

backwardstep!(state, cluster, Δts, model, calibration, G; withnegative=withnegative)
@btime backwardstep!($state, $cluster, $Δts, $model, $calibration, $G; withnegative=$withnegative)

begin
    state = interpolateovergrid(terminalstate, Gterminal, G)
    backwardsimulation!(state, model, calibration, G; verbose=1, withnegative=withnegative, tstop=400., printevery=1_000)
end

if isinteractive()
    Ffig = contourf(mspace, Tspace, state.valuefunction.Fₜ; title="log(F)")
    αfig = heatmap(Tspace, mspace, last.(state.policystate.policy); title="ε")

    tfig = heatmap(Tspace, mspace, state.timestate.t; title="t")
    focfig = heatmap(Tspace, mspace, state.policystate.foc; title="FOC")

    plot(Ffig, αfig, tfig, focfig; layout=(2, 2), size=(800, 600), margins=10Plots.mm)
end