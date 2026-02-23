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
includet("../utils/loading.jl")
includet("../plotting/utils.jl")
includet("../utils/simulating.jl")
includet("../markov/chain.jl")
includet("../markov/finitedifference.jl")
includet("../regret/chain.jl")
includet("../regret/finitedifference.jl")

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
end;

begin # Initialise the grid
    # State
    N₁ = 200; N₂ = 250;
    N = (N₁, N₂)
    Tmin = 0.; Tmax = 8.;
    decay = ConstantDecay(0.)
    linearclimate = LinearClimate(hogg, decay)
    mmin = mstable(Tmin + 0.1, linearclimate)
    mmax = mstable(Tmax - 0.1, linearclimate)
    
    Tdomain = (Tmin, Tmax)
    mdomain = (mmin, mmax)
    
    domains = (Tdomain, mdomain)
    withnegative = true

    G = RegularGrid(N, domains)

    # Time
    Δt⁻¹ = 12.
    Δt = 1 / Δt⁻¹
    τ = 500.
end;

begin
    preferences = LogSeparable()
    decay = ConstantDecay(0.)
    climate = LinearClimate(hogg, decay)

    preferences = LogSeparable()
    model = IAM(climate, economy, preferences)
end

simpath = "data/simulation-dense";
paths = loadregretpolicypaths(simpath; exclude = ["terminal"])
filteredpath = filterpolicies(paths, 10; tspan = (0., 2.)) # Indices of the basis

policybasis = SimplexPolicies(paths)