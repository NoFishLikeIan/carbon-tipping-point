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

include("utils/saving.jl")
include("markov/chain.jl")
include("markov/terminal.jl")
include("markov/backward.jl")

# Construct model
calibrationfilepath = joinpath(datapath, "calibration.jld2"); @assert isfile(calibrationfilepath)

begin
    calibrationfile = jldopen(calibrationfilepath, "r+")
    @unpack hogg, calibration, albedo = calibrationfile
    close(calibrationfile)
end

preferences = EpsteinZin(θ = rra, ψ = eis);
economy = Economy()
damages = leveldamages ? LevelDamages() : GrowthDamages()

albedo = updateTᶜ(threshold, albedo)
model = TippingModel(albedo, hogg, preferences, damages, economy)

# Construct Grid
Tdomain = hogg.Tᵖ .+ (0., 6.5);
mdomain = mstable.(Tdomain, hogg)
G = RegularGrid([Tdomain, mdomain], N)

if (verbose ≥ 1)
    println("$(now()): ","Solving tipping model with Tᶜ = $threshold, ψ = $eis, θ = $rra, $(withnegative ? "with" : "without") negative emissions and $(leveldamages ? "level" : "growth") damages...")
    flush(stdout)
end

outdir = joinpath(datapath, simulationpath, withnegative ? "negative" : "constrained")

if (verbose ≥ 1)
    println("$(now()): ","Running terminal...")
    flush(stdout)
end

computeterminal(model, G; verbose, outdir, alternate = true, tol, overwrite)

if (verbose ≥ 1)
    println("$(now()): ","Running backward...")
    flush(stdout)
end

computebackward(model, calibration, G; verbose, outdir, overwrite, tstop = stopat, cachestep, withnegative)