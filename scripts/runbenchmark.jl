using Pkg
Pkg.instantiate(); Pkg.precompile();

using UnPack: @unpack

include("arguments.jl") # Import argument parser

parsedargs = ArgParse.parse_args(argtable)

@unpack overwrite, datapath, simulationpath, N, cachestep, tol, verbose, stopat, procs, leveldamages, eis, rra, allownegative = parsedargs

overwrite && (verbose ≥ 1) && @warn "Running in overwrite mode!"

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

(verbose ≥ 1) && println("Solving jump model with ψ = $eis, θ = $rra, and $(allownegative ? "with" : "without") negative emission...")

outdir = joinpath(datapath, simulationpath, 
allownegative ? "negative" : "constrained")

(verbose ≥ 1) && println("Running terminal...")
Gterminal = terminalgrid(N, model)
computeterminal(model, Gterminal; verbose, outdir, alternate = true, tol, overwrite)

(verbose ≥ 1) && println("Running backward...")
computebackward(model, G; verbose, outdir, overwrite, tstop = stopat, cachestep, allownegative)