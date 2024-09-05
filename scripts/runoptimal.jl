include("utils/saving.jl")
include("markov/terminal.jl")
include("markov/backward.jl")

using Distributed: nprocs

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
VERBOSE && "Running with $(nprocs()) processor..."

# Parameters
thresholds = [1.5, 2.5];

# Construct model
preferences = EpsteinZin();
calibration = load_object(joinpath(DATAPATH, "calibration.jld2"));
economy = Economy()
hogg = Hogg()
damages = GrowthDamages()

# Construct Grid
Tdomain = hogg.Tᵖ .+ (0., 7.);
mdomain = mstable.(Tdomain, hogg)
G = RegularGrid([Tdomain, mdomain], N)

for Tᶜ ∈ thresholds
    VERBOSE && println("Solving model with Tᶜ = $Tᶜ and $(ALLOWNEGATIVE ? "with" : "without") negative emission...")
    
    albedo = Albedo(Tᶜ)
    model = TippingModel(albedo, hogg, preferences, damages, economy, calibration)
    
    # Terminal simulation
    if RUNTERMINAL
        Gterminal = terminalgrid(N, model)
        computeterminal(model, Gterminal; verbose = VERBOSE, datapath = datapath, alternate = true, tol = TOL, overwrite = OVERWRITE)
    end

    if RUNBACKWARDS
        VERBOSE && println("Running backward...")
        computebackward(model, G; verbose = VERBOSE, datapath = datapath, overwrite = OVERWRITE, tstop = TSTOP, cachestep = CACHESTEP, allownegative = ALLOWNEGATIVE)
    end
end
