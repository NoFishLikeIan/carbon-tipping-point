using UnPack: @unpack

include("arguments.jl") # Import argument parser

parsedargs = ArgParse.parse_args(argtable)

@unpack overwrite, datapath, simulationpath, N, cachestep, tol, verbose, stopat, procs = parsedargs

overwrite && (verbose ≥ 1) && @warn "Running in overwrite mode!"

# Distributed processing
using Distributed: nprocs, addprocs
addprocs(procs; exeflags="--project") # A bit sad that I have to do this

(verbose ≥ 1) && "Running with $(nprocs()) processor..."

include("utils/saving.jl")
include("markov/terminal.jl")
include("markov/backward.jl")

# Default parameters
thresholds = [1.5, 2.5];
Ψ = [0.75, 1.5]
Θ = [10.]
Ρ = [0., 1e-3]
Ωᵣ = [0., 0.017558043747351086]

# Construct model
calibrationdirectory = joinpath(datapath, "calibration.jld2")
calibration = load_object(calibrationdirectory);

preferences = EpsteinZin();
economy = Economy()
hogg = Hogg()
damages = GrowthDamages()

# Construct Grid
Tdomain = hogg.Tᵖ .+ (0., 9.);
mdomain = mstable.(Tdomain, hogg)
G = RegularGrid([Tdomain, mdomain], N)

for Tᶜ ∈ thresholds, allownegative in [false, true]
    (verbose ≥ 1) && println("Solving model with Tᶜ = $Tᶜ and $(allownegative ? "with" : "without") negative emission...")

    outdir = joinpath(datapath, simulationpath, 
        allownegative ? "negative" : "constrained")
    
    albedo = Albedo(Tᶜ)
    model = TippingModel(albedo, hogg, preferences, damages, economy, calibration)
    
    (verbose ≥ 1) && println("Running terminal...")
    Gterminal = terminalgrid(N, model)
    computeterminal(model, Gterminal; verbose, outdir, alternate = true, tol, overwrite)

    (verbose ≥ 1) && println("Running backward...")
    computebackward(model, G; verbose, outdir, overwrite, tstop = stopat, cachestep, allownegative)
end
