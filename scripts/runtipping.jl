using Pkg
Pkg.resolve(); Pkg.instantiate();

using Base.Threads: nthreads
using UnPack: @unpack
using Dates: now

include("arguments.jl") # Import argument parser

parsedargs = ArgParse.parse_args(argtable)

@unpack overwrite, datapath, simulationpath, N, cachestep, tol, verbose, stopat, threshold, leveldamages, eis, rra, withnegative = parsedargs

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
using LinearAlgebra: norm

using Optimization, OptimizationOptimJL
using ForwardDiff

using JLD2
using Printf, Dates

include("../src/valuefunction.jl")
include("../src/extensions.jl")
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
damages = leveldamages ? WeitzmanLevel() : Kalkuhl()

feedback = Model.updateTᶜ(threshold + hogg.Tᵖ, feedback)
model = TippingModel(hogg, preferences, damages, economy, feedback)

# Construct Grid
Tdomain = hogg.Tᵖ .+ (0., 6.);
mdomain = mstable.(Tdomain, hogg)
G = RegularGrid((Tdomain, mdomain), N, hogg)

if (verbose ≥ 1)
    println("$(now()): ","Solving tipping model with Tᶜ = $threshold, ψ = $eis, θ = $rra, $(withnegative ? "with" : "without") negative emissions and $(leveldamages ? "level" : "growth") damages...")
    flush(stdout)
end

outdir = joinpath(datapath, simulationpath, withnegative ? "negative" : "constrained")

if (verbose ≥ 1)
    println("$(now()): ","Running terminal...")
    flush(stdout)
end

state, Gterminal = computeterminal(model, calibration, G; verbose, outdir, alternate = true, tol, overwrite)

if (verbose ≥ 1)
    println("$(now()): ","Running backward...")
    flush(stdout)
end

state = computebackward((state, Gterminal), model, calibration, G; verbose, outdir, overwrite, tstop = stopat, cachestep, withnegative)