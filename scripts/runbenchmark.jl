include("utils/saving.jl")
include("markov/terminal.jl")
include("markov/backward.jl")

N = getnumber(env, "N", 51; type = Int)

VERBOSE = getbool(env, "VERBOSE", false)
RUNTERMINAL = getbool(env, "RUNTERMINAL", false)
RUNBACKWARDS = getbool(env, "RUNBACKWARDS", false)
OVERWRITE = getbool(env, "OVERWRITE", false)
TOL = getnumber(env, "TOL", 1e-3)
TSTOP = getnumber(env, "TSTOP", 0.)
CACHESTEP = getnumber(env, "CACHESTEP", 1 / 4)

# Construct model
preferences = EpsteinZin();
calibration = load_object(joinpath(DATAPATH, "calibration.jld2"));
economy = Economy()
hogg = Hogg()
damages = [GrowthDamages()]
negativeemissions = [false]

# Terminal simulation
for d in damages
    VERBOSE && println("\nSolving for damages = $d...")
    
    jumpmodel = JumpModel(Jump(), hogg, preferences, d, economy, calibration)

    G = constructdefaultgrid(N, jumpmodel)
    VERBOSE && println("\nSolving jump model...")
    if RUNTERMINAL
        computeterminal(jumpmodel, G; verbose = VERBOSE, datapath = DATAPATH, alternate = true, tol = TOL)
    end

    if RUNBACKWARDS
        for allownegative in negativeemissions
            VERBOSE && println("Running backward $(ifelse(allownegative, "with", "without")) negative emissions...")
            computebackward(jumpmodel, G; allownegative, verbose = VERBOSE, datapath = DATAPATH, overwrite = OVERWRITE, tstop = TSTOP, cachestep = CACHESTEP)
        end
    end
end
