using Pkg
Pkg.resolve(); Pkg.instantiate();

using Base.Threads: nthreads
using UnPack: @unpack
using Dates: now

include("arguments.jl") # Import argument parser

parsedargs = ArgParse.parse_args(argtable)

@unpack datapath, simulationpath, overwrite = parsedargs # File system parameters
@unpack cachestep, verbose, stopat = parsedargs # IO parameters
@unpack NT, Nm, tol, dt = parsedargs # Simulation parameters
@unpack threshold, damages, eis, rra, withnegative = parsedargs # Problem parameters

if !(eis ≈ 1)
    throw("Case ψ ≠ 1 not implemented yet!")
end

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
using Statistics
using StaticArrays, SparseArrays
using LinearSolve, LinearAlgebra

using Optimization, OptimizationOptimJL, LineSearches
using ForwardDiff

using JLD2
using Printf, Dates

include("../src/valuefunction.jl")
include("../src/extend/model.jl")
include("../src/extend/valuefunction.jl")
include("utils/saving.jl")
include("markov/utils.jl")
include("markov/chain.jl")
include("markov/finitedifference.jl")

# Construct model
begin
    calibrationfilepath = "data/calibration.jld2"; @assert isfile(calibrationfilepath)

	calibrationfile = jldopen(calibrationfilepath, "r+")
	@unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = calibrationfile
	close(calibrationfile)
end

preferences = Preferences(θ = rra, ψ = 1.0);
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
Tmin, Tmax = hogg.Tᵖ .+ (0., 8.5);
Tdomain = (Tmin, Tmax)
mdomain = mstable(Tmin + 0.5, model), mstable(Tmax - 0.5, model)
N = (NT, Nm)
Gterminal = RegularGrid(N, (Tdomain, mdomain))

if (verbose ≥ 1)
    modelstring = model isa TippingModel ? "tipping model with Tᶜ = $threshold," : "linear model with"

    println("$(now()): ","Solving $modelstring ψ = $eis, θ = $rra, $(withnegative ? "with" : "without") negative emissions and $damages damages...")
    flush(stdout)
end

outdir = joinpath(datapath, simulationpath)

if (verbose ≥ 1)
    println("$(now()): ","Running terminal...")
    flush(stdout)
end

tolerance = Error(tol, 1e-3)
terminalvaluefunction = ValueFunction(hogg, Gterminal, calibration)
steadystate!(terminalvaluefunction, dt, model, Gterminal, calibration; verbose, tolerance, withnegative)

if (verbose ≥ 1)
    println("$(now()): ","Running backward...")
    flush(stdout)
end

G = shrink(Gterminal, 0.1)
valuefunction = interpolateovergrid(terminalvaluefunction, Gterminal, G)
backwardsimulation!(valuefunction, dt, model, G, calibration; verbose, withnegative, overwrite, outdir, cachestep = cachestep)