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
    N₁ = 30; N₂ = 31;
    N = (N₁, N₂)
    Tmin = 0.; Tmax = 8.;
    mmin = mstable(Tmin + 0.1, model.climate)
    mmax = mstable(Tmax - 0.1, model.climate)
    
    Tdomain = (Tmin, Tmax)
    mdomain = (mmin, mmax)
    
    domains = (Tdomain, mdomain)
    withnegative = true

    G = RegularGrid(N, domains)
    Δt⁻¹ = 24.
    Δt = 1 / Δt⁻¹
    τ = 500.
end;

valuefunction = ValueFunction(τ, climate, G, calibration)

begin
    tspace = 0:0.1:τ
    Amat =  [max(m - 0.5, 0.) * γ(t, calibration) for T in G.ranges[1], m in G.ranges[2], t in tspace]
    abatement =  linear_interpolation((G.ranges[1], G.ranges[2], tspace), Amat; extrapolation_bc = Interpolations.Flat())
    
    exogenousvaluefunction = copy(valuefunction)
    exogenoussteadystate!(exogenousvaluefunction, Δt, model, G, calibration; verbose = true)

    exogenousvaluefunctiontraj = exogenousbackwardsimulation!(exogenousvaluefunction, abatement, Δt, model, G, calibration; t₀ = 0., verbose = 1, printstep = 10, withsave = false, cachestep = 1., storetrajectory = false)
end

begin
    endogenousvaluefunction = copy(valuefunction)

    exogenoussteadystate!(endogenousvaluefunction, Δt, model, G, calibration; verbose = true)

    endogenousvaluefunctiontraj = backwardsimulation!(endogenousvaluefunction, Δt, model, G, calibration; t₀ = 0., verbose = 1, printstep = 10, withsave = false, cachestep = 1., storetrajectory = false, withnegative = true)
end

k = minimum(keys(exogenousvaluefunctiontraj))
R =  exogenousvaluefunctiontraj[k].H .- endogenousvaluefunctiontraj[k].H


Δα = endogenousvaluefunctiontraj[k].α .- exogenousvaluefunctiontraj[k].α
