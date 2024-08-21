include("utils/saving.jl")
include("markov/terminal.jl")
include("markov/backward.jl")

env = DotEnv.config()
DATAPATH = get(env, "DATAPATH", "data")
SIMPATH = get(env, "SIMULATIONPATH", "simulation/planner")
datapath = joinpath(DATAPATH, SIMPATH)

N = getnumber(env, "N", 31; type = Int)
VERBOSE = getbool(env, "VERBOSE", false)
RUNTERMINAL = getbool(env, "RUNTERMINAL", false)
RUNBACKWARDS = getbool(env, "RUNBACKWARDS", false)
OVERWRITE = getbool(env, "OVERWRITE", false)
TOL = getnumber(env, "TOL", 1e-3)
TSTOP = getnumber(env, "TSTOP", 0.)
CACHESTEP = getnumber(env, "CACHESTEP", 1 / 4)

OVERWRITE && @warn "Running in overwrite mode!"

# Construct model
preferences = EpsteinZin();
calibration = load_object(joinpath(DATAPATH, "calibration.jld2"));
economy = Economy()
hogg = Hogg()
damages = GrowthDamages()
jump = Jump()
allownegative = false

jumpmodel = JumpModel(jump, hogg, preferences, damages, economy, calibration)

# Construct Grid
Tdomain = hogg.Tᵖ .+ (0., 7.);
mdomain = (mstable(Tdomain[1], hogg), mstable(Tdomain[2], hogg))
G = RegularGrid([Tdomain, mdomain], N)

VERBOSE && println("\nSolving jump model...")
if RUNTERMINAL
    Gterminal = terminalgrid(N, jumpmodel)
    computeterminal(jumpmodel, Gterminal; verbose = VERBOSE, datapath = datapath, alternate = true, tol = TOL)
end

if RUNBACKWARDS
    VERBOSE && println("Running backward $(ifelse(allownegative, "with", "without")) negative emissions...")
    computebackward(jumpmodel, G; allownegative, verbose = VERBOSE, datapath = datapath, overwrite = OVERWRITE, tstop = TSTOP, cachestep = CACHESTEP)
end