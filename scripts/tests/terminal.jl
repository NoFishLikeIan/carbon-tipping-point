using Test, BenchmarkTools, Revise
using Plots, LaTeXStrings, ColorSchemes

using Model, Grid
using Base.Threads
using SciMLBase
using Statistics
using StaticArrays, SparseArrays
using LinearSolve, LinearAlgebra

using JLD2, UnPack
using Dates, Printf

includet("../../src/valuefunction.jl")
includet("../../src/extend/model.jl")
includet("../../src/extend/grid.jl")
includet("../../src/extend/valuefunction.jl")
includet("../utils/saving.jl")
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
    damages = WeitzmanGrowth() # NoDamageGrowth{Float64}()
    economy = Economy(investments = investments, damages = damages, abatement = abatement)

    # Load climate claibration
    climatepath = joinpath(calibrationpath, "climate.jld2")
    @assert isfile(climatepath) "Climate calibration file not found at $climatepath"
    climatefile = jldopen(climatepath, "r+")
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher, decay = climatefile
    close(climatefile)

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
end

begin
    N₁ = 71; N₂ = 71;
    N = (N₁, N₂)
    Tmin = 0.; Tmax = 10.;
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

equilibriumsteadystate!(valuefunction, Δt, linearIAM(model), G, calibration; verbose = 2, timeiterations = 100_000, printstep = 10_000, tolerance = Error(1e-8, 1e-8))
eqvaluefunction = deepcopy(valuefunction)

steadystate!(valuefunction, Δt, model, G, calibration; timeiterations = 10_000, printstep = 1_000, verbose = 1, tolerance = Error(1e-7, 1e-8))