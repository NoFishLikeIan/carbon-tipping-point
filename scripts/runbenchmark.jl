using Pkg
Pkg.resolve(); Pkg.instantiate();

using UnPack: @unpack
using Dates: now
using Base.Threads: nthreads

include("arguments.jl") # Import argument parser

parsedargs = ArgParse.parse_args(argtable)

@unpack overwrite, datapath, simulationpath, N, cachestep, tol, verbose, stopat, leveldamages, eis, rra, allownegative = parsedargs

if (verbose ≥ 1)
    println("$(now()): ", "Running with $(nthreads()) threads...")

    if overwrite
        println("$(now()): ", "Running in overwrite mode!")
    end
    
    flush(stdout)
end

include("utils/saving.jl")
include("markov/terminal.jl")
include("markov/backward.jl")

# Construct model
calibrationdirectory = joinpath(datapath, "calibration.jld2")
calibration = load_object(calibrationdirectory);

hogg = Hogg()
preferences = EpsteinZin(θ = rra, ψ = eis);
economy = Economy()
damages = leveldamages ? LevelDamages() : GrowthDamages()

jump = Jump()
model = JumpModel(jump, hogg, preferences, damages, economy, calibration)

# Construct Grid
Tdomain = hogg.Tᵖ .+ (0., 9.);
mdomain = mstable.(Tdomain, hogg)
G = RegularGrid([Tdomain, mdomain], N)

if (verbose ≥ 1)
    println("$(now()): ","Solving benchmark model with ψ = $eis, θ = $rra, and $(allownegative ? "with" : "without") negative emission and $(leveldamages ? "level" : "growth") damages...")
    flush(stdout)
end

outdir = joinpath(datapath, simulationpath, allownegative ? "negative" : "constrained")

if (verbose ≥ 1)
    println("$(now()): ","Running terminal...")
    flush(stdout)
end

Gterminal = terminalgrid(N, model)
computeterminal(model, Gterminal; verbose, outdir, alternate = true, tol, overwrite)

if (verbose ≥ 1)
    println("$(now()): ","Running backward...")
    flush(stdout)
end

# TODO: Test parallelisation
computebackward(model, calibration, G; verbose, outdir, overwrite, tstop = stopat, cachestep, allownegative)