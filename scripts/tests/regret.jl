using Test, BenchmarkTools, Revise, UnPack
using Plots, LaTeXStrings
default(c=:viridis, label=false, dpi=180)

using Model, Grid
using Base.Threads
using SciMLBase
using Statistics
using StaticArrays, SparseArrays
using Interpolations, DataStructures

using LinearSolve, LinearAlgebra

using JLD2, UnPack
using Dates, Printf

includet("../../src/valuefunction.jl")
includet("../../src/extend/model.jl")
includet("../../src/extend/grid.jl")
includet("../../src/extend/valuefunction.jl")
includet("../utils/saving.jl")
includet("../utils/simulating.jl")
includet("../plotting/utils.jl")
includet("../markov/utils.jl")
includet("../markov/chain.jl")
includet("../markov/finitedifference.jl")

begin # Construct the model
    DATAPATH = "data"
    calibrationpath = joinpath(DATAPATH, "calibration")

    # Load economic calibration
    abatementpath = joinpath(calibrationpath, "abatement.jld2")
    @assert isfile(abatementpath) "Abatement calibration file not found at $abatementpath"
    abatementfile = jldopen(abatementpath, "r+")
    @unpack abatement = abatementfile
    close(abatementfile)

    investments = Investment()
    damages = BurkeHsiangMiguel() # WeitzmanGrowth()
    economy = Economy(investments = investments, damages = damages, abatement = abatement)

    # Load climate claibration
    climatepath = joinpath(calibrationpath, "climate.jld2")
    @assert isfile(climatepath) "Climate calibration file not found at $climatepath"
    climatefile = jldopen(climatepath, "r+")
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = climatefile
    close(climatefile)

    decay = ConstantDecay(0.)
    threshold = 2.
    climate = if 0 < threshold < Inf
        feedback = Model.updateTᶜ(threshold, feedback)
        TippingClimate(hogg, decay, feedback)
    else
        LinearClimate(hogg, decay)
    end

    preferences = LogSeparable()
    model = IAM(climate, economy, preferences)
    model = determinsticIAM(model)
end;

begin # Initialise the grid
    N₁ = 50; N₂ = 51;
    N = (N₁, N₂)
    Tmin = 0.; Tmax = 8.;
    mmin = mstable(Tmin + 0.1, model.climate)
    mmax = mstable(Tmax - 0.1, model.climate)
    
    Tdomain = (Tmin, Tmax)
    mdomain = (mmin, mmax)
    
    domains = (Tdomain, mdomain)
    withnegative = true

    G = RegularGrid(N, domains)
    Δt⁻¹ = 12.
    Δt = 1 / Δt⁻¹
    τ = 500.
end;

valuefunction = ValueFunction(τ, climate, G, calibration)

begin
    endvaluefunction = copy(valuefunction)

    steadystate!(endvaluefunction, Δt, model, G, calibration; verbose = true)

    endvaluefunctiontraj = backwardsimulation!(endvaluefunction, Δt, model, G, calibration; t₀ = 0., verbose = 1, printstep = 10, withsave = false, cachestep = 1., storetrajectory = false, withnegative = true)
end

begin
    tspace = 0:τ
    Amat =  [max(m - 0.5, 0.) * γ(t, calibration) for T in G.ranges[1], m in G.ranges[2], t in tspace]
    abatement =  linear_interpolation((G.ranges[1], G.ranges[2], tspace), Amat; extrapolation_bc = Interpolations.Flat())
    
    exvaluefunction = copy(valuefunction)
    setpolicy!(exvaluefunction, abatement, G)
    exogenoussteadystate!(exvaluefunction, Δt, model, G, calibration; verbose = true)

    exvaluefunctiontraj = exogenousbackwardsimulation!(exvaluefunction, abatement, Δt, model, G, calibration; t₀ = 0., verbose = 1, printstep = 10, withsave = false, cachestep = 1., storetrajectory = false)
end

firstidx, lastidx = extrema(keys(exvaluefunctiontraj))

begin
    R̄ = exvaluefunctiontraj[lastidx].H - 
        endvaluefunctiontraj[lastidx].H;
        
    lastfig = contourf(G.ranges[2], G.ranges[1], R̄; c = :Reds, ylabel = L"Temperature $T\degree$", xlabel = L"Log-$\textrm{CO}_2\textrm{e}$ concentration $m$", clims = (0, Inf), linewidth = 0, title = L"Terminal regret $\overline{R}$")

    R₀ = exvaluefunctiontraj[firstidx].H - 
        endvaluefunctiontraj[firstidx].H;

    firstfig = contourf(G.ranges[2], G.ranges[1], R₀; c = :Reds, ylabel = L"Temperature $T\degree$", xlabel = L"Log-$\textrm{CO}_2\textrm{e}$ concentration $m$", clims = (0, Inf), linewidth = 0, title = L"Initial regret $R_0$")

    plot(firstfig, lastfig; size = 400 .* (2√2, 1), margins = 5Plots.mm)
end