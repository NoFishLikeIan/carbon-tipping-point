using UnPack: @unpack

include("arguments.jl") # Import argument parser

parsedargs = ArgParse.parse_args(argtable)

@unpack overwrite, datapath, simulationpath, N, cachestep, tol, verbose, stopat, procs = parsedargs

# Distributed processing
using Distributed: nprocs, addprocs
addprocs(procs; exeflags="--project") # A bit sad that I have to do this

(verbose ≥ 1) && "Running with $(nprocs()) processor..."

include("utils/saving.jl")
include("markov/terminal.jl")
include("markov/backward.jl")

env = DotEnv.config()
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

# Parameters
Ψ = [0.75, 1.5]
Θ = [2., 10.]
Ρ = [1e-7, 1e-3]
Ωᵣ = [0., 0.017558043747351086]

# Construct model
calibration = load_object(joinpath(DATAPATH, "calibration.jld2"));
hogg = Hogg()
damages = GrowthDamages()
jump = Jump()

model = JumpModel(jump, hogg, preferences, damages, economy, calibration)

# Construct Grid
Tdomain = hogg.Tᵖ .+ (0., 9.);
mdomain = mstable.(Tdomain, hogg)
G = RegularGrid([Tdomain, mdomain], N)

for Tᶜ ∈ thresholds, ψ ∈ Ψ, θ ∈ Θ, ϱ ∈ Ρ, ωᵣ ∈ Ωᵣ
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