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
includet("../markov/certaintyequivalence.jl")
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
    truemodel = IAM(climate, economy, preferences)
    linearmodel = linearIAM(truemodel)
end

discovery = -2.
linearsimulation = loadtotal(linearmodel; outdir = joinpath(DATAPATH, "simulation-dense"))
truesimulation = loadtotal(truemodel; outdir = joinpath(DATAPATH, "simulation-dense"))

values, model, G = discoveryvalues(discovery, truesimulation, linearsimulation)
_, αitp = buildinterpolations(values, G);

τ = maximum(keys(values))
valuefunction = copy(values[τ])
Δt = 0.1
staticbackward!(valuefunction, Δt, αitp, model, G, calibration; t₀ = 0., verbose = 2, alg = KLUFactorization())

# Compare to optimal value function
optimalvaluefunction = truesimulation[1][0.0]