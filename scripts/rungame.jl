using Pkg
Pkg.resolve(); Pkg.instantiate();

using Base.Threads: nthreads
using UnPack: @unpack
using Dates: now

include("arguments.jl") # Import argument parser

parsedargs = ArgParse.parse_args(argtable)

@unpack overwrite, datapath, simulationpath, N, cachestep, tol, verbose, stopat, threshold, eis, rra, allownegative = parsedargs

if (verbose ≥ 1)
    println("$(now()): ", "Running with $(nthreads()) threads...")

    if overwrite
        println("$(now()): ", "Running in overwrite mode!")
    end

    flush(stdout)
end

# Begin script
using JLD2
using Model, Grid

include("utils/saving.jl")
include("markov/terminal.jl")
include("markov/game.jl")


# Construct model
# -- Calibration
calibration = load_object(joinpath(datapath, "calibration.jld2"));
regionalcalibration = load_object(joinpath(datapath, "regionalcalibration.jld2"));
regionalcalibrations = [regionalcalibration[:oecd], regionalcalibration[:row]]

# -- Climate
hogg = Hogg()
albedo = Albedo(threshold)

# -- Economy and Preferences
preferences = EpsteinZin(θ = rra, ψ = eis);
oecdeconomy, roweconomy = RegionalEconomies()

oecdmodel = TippingModel(albedo, hogg, preferences, LevelDamages(), oecdeconomy)
rowmodel = TippingModel(albedo, hogg, preferences, GrowthDamages(), roweconomy)
models = [oecdmodel, rowmodel]

# Construct Grid
Tdomain = hogg.Tᵖ .+ (0., 6.);
mdomain = mstable.(Tdomain, hogg)
G = RegularGrid([Tdomain, mdomain], N)

if (verbose ≥ 1)
    println("$(now()): ","Solving game model with Tᶜ = $threshold, ψ = $eis, θ = $rra...")
    flush(stdout)
end

outdir = joinpath(datapath, simulationpath)

if (verbose ≥ 1)
    println("$(now()): ","Running terminal...")
    flush(stdout)
end

Gterminal = terminalgrid(N, oecdmodel)

begin # Terminal problem
	computeterminal(oecdmodel, G; verbose, outdir, addpath = "oecd", alternate = true, tol, overwrite)
	computeterminal(rowmodel, G; verbose, outdir, addpath = "row", alternate = true, tol, overwrite)
end

if (verbose ≥ 1)
    println("$(now()): ","Running backward...")
    flush(stdout)
end

computebackward(models, regionalcalibrations, calibration, G; verbose, outdir, overwrite, tstop = stopat, cachestep, allownegative, addpaths = ["oecd", "row"])