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
SMOOTHING = getnumber(env, "SMOOTHING", 1 / 2)

OVERWRITE && @warn "Running in overwrite mode!"

# Construct model
thresholds = [1.5, 1.8, 2.5, 3., 3.5];
preferences = EpsteinZin();
calibration = load_object(joinpath(DATAPATH, "calibration.jld2"));
economy = Economy()
hogg = Hogg()
damages = [GrowthDamages()]
negativeemissions = [false]

# Terminal simulation
for d in damages
    VERBOSE && println("\nSolving for damages = $d...")
    
    for Tᶜ ∈ thresholds
        VERBOSE && println("Solving model with Tᶜ = $Tᶜ...")

        albedo = Albedo(Tᶜ = Tᶜ)
        model = TippingModel(albedo, preferences, d, economy, hogg, calibration)

        G = constructdefaultgrid(N, model)

        RUNTERMINAL && computeterminal(model, G; verbose = VERBOSE, datapath = DATAPATH, alternate = true, tol = TOL, overwrite = OVERWRITE)

        if RUNBACKWARDS
            for allownegative in negativeemissions
                VERBOSE && println("Running backward $(ifelse(allownegative, "with", "without")) negative emissions...")
                computebackward(model, G; allownegative, verbose = VERBOSE, datapath = DATAPATH, overwrite = OVERWRITE, tstop = TSTOP, cachestep = CACHESTEP)
            end
        end
    end
end
