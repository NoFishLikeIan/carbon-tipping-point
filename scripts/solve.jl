using Pkg
Pkg.resolve(); Pkg.instantiate();

using Base.Threads: nthreads
using UnPack: @unpack
using Dates: now

include("arguments.jl") # Import argument parser

parsedargs = ArgParse.parse_args(argtable)

@unpack datapath, simulationpath, overwrite = parsedargs # File system parameters
@unpack cachestep, verbose, stopat = parsedargs # IO parameters
@unpack N, tol = parsedargs # Simulation parameters
@unpack threshold, damages, eis, rra, withnegative = parsedargs # Problem parameters


if (verbose ≥ 1)
    println("$(now()): ", "Running with $(nthreads()) threads...")

    if overwrite
        println("$(now()): ", "Running in overwrite mode!")
    end
    flush(stdout)
end

# Begin script
using Model, Grid

using Base.Threads

using SciMLBase
using ZigZagBoomerang
using Statistics
using StaticArrays
using FastClosures
using LinearAlgebra

using Optimization, OptimizationOptimJL, LineSearches
using ForwardDiff

using JLD2
using Printf, Dates

include("../src/valuefunction.jl")
include("../src/extend/model.jl")
include("../src/extend/grid.jl")
include("utils/saving.jl")
include("utils/logging.jl")
include("markov/chain.jl")
include("markov/terminal.jl")
include("markov/backward.jl")

# Construct model
begin
    calibrationfilepath = "data/calibration.jld2"; @assert isfile(calibrationfilepath)

	calibrationfile = jldopen(calibrationfilepath, "r+")
	@unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = calibrationfile
	close(calibrationfile)
end

preferences = Preferences(θ = rra, ψ = eis);
economy = Economy()
damage = if damages == "kalkuhl"
    Kalkuhl{Float64}()
elseif damages == "nodamages"
    NoDamageGrowth{Float64}()
elseif damages == "weitzman"
    WeitzmanGrowth{Float64}()
else
    error("Unknown damage type: $damages")
end

model = if threshold > 0
    feedback = Model.updateTᶜ(threshold + hogg.Tᵖ, feedback)
    TippingModel(hogg, preferences, damage, economy, feedback)
else
    LinearModel(hogg, preferences, damage, economy)
end

# Construct Grid
Tdomain = hogg.Tᵖ .+ (0., 8.5);
mdomain = mstable.(Tdomain, model)
Gterminal = constructgrid((Tdomain, mdomain), N, hogg)

if (verbose ≥ 1)
    modelstring = model isa TippingModel ? "tipping model with Tᶜ = $threshold," : "linear model with"

    println("$(now()): ","Solving $modelstring ψ = $eis, θ = $rra, $(withnegative ? "with" : "without") negative emissions and $damages damages...")
    flush(stdout)
end

outdir = joinpath(datapath, simulationpath, withnegative ? "negative" : "constrained")

if (verbose ≥ 1)
    println("$(now()): ","Running terminal...")
    flush(stdout)
end

state, Gterminal = computeterminal(model, calibration, Gterminal; verbose = verbose, outdir = outdir, alternate = true, tol = tol, overwrite = overwrite)

if (verbose ≥ 1)
    println("$(now()): ","Running backward...")
    flush(stdout)
end

G = shrink(Gterminal, 0.9, hogg)
state = computebackward((state, Gterminal), model, calibration, G; verbose = verbose, outdir = outdir, overwrite = overwrite, tstop = stopat, cachestep = cachestep, withnegative = withnegative)