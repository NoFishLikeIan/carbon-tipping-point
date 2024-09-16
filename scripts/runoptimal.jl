using DotEnv: config
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

# Parameters
thresholds = [1.5, 2.5];
Ψ = [0.75, 1.5]
Θ = [10.]
Ρ = [0., 1e-3]
Ωᵣ = [0., 0.017558043747351086]

# Construct model
calibration = load_object(joinpath(DATAPATH, "calibration.jld2"));
hogg = Hogg()
damages = GrowthDamages()

# Construct Grid
Tdomain = hogg.Tᵖ .+ (0., 9.);
mdomain = mstable.(Tdomain, hogg)
G = RegularGrid([Tdomain, mdomain], N)

for Tᶜ ∈ thresholds, ψ ∈ Ψ, θ ∈ Θ, ϱ ∈ Ρ, ωᵣ ∈ Ωᵣ
    preferences = EpsteinZin(θ = θ, ψ = ψ);
    economy = Economy(ϱ = ϱ, ωᵣ = ωᵣ)

    VERBOSE && println("Solving model with Tᶜ = $Tᶜ, ψ = $ψ, θ = $θ, ϱ = $ϱ, ωᵣ = $ωᵣ, and $(ALLOWNEGATIVE ? "with" : "without") negative emission...")
    
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
