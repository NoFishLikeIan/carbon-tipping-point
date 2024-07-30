include("utils/saving.jl")
include("terminal.jl")
include("backward.jl")

ΔΛ = [0., 0.06, 0.08];
N = 51;

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
damages = (GrowthDamages(), ) # (LevelDamages(), GrowthDamages())
negativeemissions = (false, true)

# Terminal simulation
for d in damages
    if VERBOSE
        println("Solving for damages = $d...")
    end
    
    for Δλ ∈ ΔΛ
        VERBOSE && println("Solving albedo model Δλ = $Δλ")

        λ₂ = Albedo().λ₁ - Δλ
        albedo = Albedo(λ₂ = λ₂)
        model = TippingModel(albedo, preferences, d, economy, hogg, calibration)

        G = constructdefaultgrid(N, model)

        RUNTERMINAL && computeterminal(model, G; verbose = VERBOSE, datapath = DATAPATH, alternate = true, tol = TOL)

        if RUNBACKWARDS
            for allownegative in negativeemissions
                VERBOSE && println("Running backward $(ifelse(allownegative, "with", "without")) negative emissions...")
                computebackward(model, G; allownegative, verbose = VERBOSE, datapath = DATAPATH, overwrite = OVERWRITE, tstop = TSTOP, cachestep = CACHESTEP)
            end
        end
    end

    jumpmodel = JumpModel(Jump(), preferences, d, economy, hogg, calibration)

    G = constructdefaultgrid(N, jumpmodel)

    VERBOSE && println("Solving jump model")

    RUNTERMINAL && computeterminal(jumpmodel, G; verbose = VERBOSE, datapath = DATAPATH, alternate = true, tol = TOL)
    if RUNBACKWARDS
        for allownegative in negativeemissions
            VERBOSE && println("Running backward $(ifelse(allownegative, "with", "without")) negative emissions...")
            computebackward(jumpmodel, G; allownegative, verbose = VERBOSE, datapath = DATAPATH, overwrite = OVERWRITE, tstop = TSTOP, cachestep = CACHESTEP)
        end
    end
end
