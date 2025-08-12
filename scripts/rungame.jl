using Pkg
Pkg.resolve(); Pkg.instantiate();

using Base.Threads: nthreads
using UnPack: @unpack
using Dates: now

include("arguments.jl") # Import argument parser

parsedargs = ArgParse.parse_args(argtable)

@unpack overwrite, datapath, simulationpath, N, cachestep, tol, verbose, stopat, threshold, eis, rra = parsedargs

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
regionalcalibration = load_object(joinpath(datapath, "regionalcalibration.jld2"));

# -- Climate
hogg = Hogg()

# -- Economy and Preferences
preferences = EpsteinZin(θ = rra, ψ = eis);
oecdeconomy, roweconomy = RegionalEconomies()
damages = Kalkuhl()

oecdmodel = LinearModel(hogg, preferences, damages, oecdeconomy)

rowmodel = threshold > 0. ?
    TippingModel(Albedo(threshold), hogg, preferences, damages, roweconomy) :
    LinearModel(hogg, preferences, damages, roweconomy)


models = AbstractModel[oecdmodel, rowmodel]

# Construct Grid
Tdomain = hogg.Tᵖ .+ (0., 6.);
mdomain = mstable.(Tdomain, hogg)
G = RegularGrid([Tdomain, mdomain], N)

if (verbose ≥ 1)
    thresholdprint = threshold > 0. ? "Tᶜ = $threshold" : "linear climate"

    println("$(now()): ","Solving game model with $thresholdprint, ψ = $eis, θ = $rra...")
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

computebackward(models, regionalcalibration, G; verbose, outdir, overwrite, tstop = stopat, cachestep, addpaths = ["oecd", "row"])