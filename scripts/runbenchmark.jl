using UnPack: @unpack

include("arguments.jl") # Import argument parser

parsedargs = ArgParse.parse_args(argtable)

@unpack overwrite, datapath, simulationpath, N, cachestep, tol, verbose, stopat, procs = parsedargs

# Distributed processing
using Distributed: nprocs, addprocs
addprocs(procs; exeflags="--project") # A bit sad that I have to do this

(verbose ≥ 1) && "Running with $(nprocs()) processor..."

include("utils/saving.jl")

# Setup environment via .env
env = config()

DATAPATH = get(env, "DATAPATH", "data")
SIMPATH = get(env, "SIMULATIONPATH", "simulation/planner")
ALLOWNEGATIVE = getbool(env, "ALLOWNEGATIVE", false)

datapath = joinpath(DATAPATH, SIMPATH, ALLOWNEGATIVE ? "negative" : "")

N = getnumber(env, "N", 31; type = Int)
VERBOSE = getbool(env, "VERBOSE", false)
RUNTERMINAL = getbool(env, "RUNTERMINAL", false)
RUNBACKWARDS = getbool(env, "RUNBACKWARDS", false)
OVERWRITE = getbool(env, "OVERWRITE", false)
TOL = getnumber(env, "TOL", 1e-3)
TSTOP = getnumber(env, "TSTOP", 0.)
CACHESTEP = getnumber(env, "CACHESTEP", 1 / 4)

OVERWRITE && @warn "Running in overwrite mode!"

# Distributed processing
using Distributed: nprocs, addprocs
ADDPROCS = getnumber(env, "ADDPROCS", 0; type = Int)
addprocs(ADDPROCS; exeflags="--project") # A bit sad that I have to do this

VERBOSE && "Running with $(nprocs()) processor..."

include("markov/terminal.jl")
include("markov/backward.jl")

# Construct model
calibration = load_object(joinpath(DATAPATH, "calibration.jld2"));
hogg = Hogg()
damages = GrowthDamages()
jump = Jump()

# Construct Grid
Tdomain = hogg.Tᵖ .+ (0., 9.);
mdomain = mstable.(Tdomain, hogg)
G = RegularGrid([Tdomain, mdomain], N)

for ψ ∈ Ψ, θ ∈ Θ, ϱ ∈ Ρ, ωᵣ ∈ Ωᵣ
    preferences = EpsteinZin(θ = θ, ψ = ψ);
    economy = Economy(ϱ = ϱ, ωᵣ = ωᵣ)
    jumpmodel = JumpModel(jump, hogg, preferences, damages, economy, calibration)


    VERBOSE && println("\nSolving jump model $(ALLOWNEGATIVE ? "with" : "without") negative emission...")
    if RUNTERMINAL
        Gterminal = terminalgrid(N, jumpmodel)
        computeterminal(jumpmodel, Gterminal; verbose = VERBOSE, datapath = datapath, alternate = true, tol = TOL, overwrite = OVERWRITE)
    end

    if RUNBACKWARDS
        VERBOSE && println("Running backward simulation...")
        computebackward(jumpmodel, G; verbose = VERBOSE, datapath = datapath, overwrite = OVERWRITE, tstop = TSTOP, cachestep = CACHESTEP)
    end
end