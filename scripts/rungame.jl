include("utils/saving.jl")
include("markov/terminal.jl")

env = DotEnv.config(".envgame")
DATAPATH = get(env, "DATAPATH", "data")
SIMPATH = get(env, "SIMULATIONPATH", "game")

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
begin
    thresholds = [1.5, 1.8, 2.5, 3., 3.5];
    preferences = EpsteinZin();
    rc = load_object(joinpath(DATAPATH, "regionalcalibration.jld2"));
    economies = RegionalEconomies()
    hogg = Hogg()
    damages = GrowthDamages()
end

# Terminal simulation
for Tᶜ ∈ thresholds
    VERBOSE && println("Solving model with Tᶜ = $Tᶜ...")

    albedo = Albedo(Tᶜ = Tᶜ)
    model = TippingGameModel(albedo, hogg, (preferences, preferences), (damages, damages), economies, rc)

    G = constructdefaultgrid(N, model)

    highmodel, lowmodel = breakgamemodel(model)

    if RUNTERMINAL
        computeterminal(highmodel, G; verbose = VERBOSE, datapath = joinpath(DATAPATH, SIMPATH, "high"), alternate = true, tol = TOL, overwrite = OVERWRITE)

        computeterminal(lowmodel, G; verbose = VERBOSE, datapath = joinpath(DATAPATH, SIMPATH, "low"), alternate = true, tol = TOL, overwrite = OVERWRITE)
    end

    if RUNBACKWARDS
        for allownegative in negativeemissions
            VERBOSE && println("Running backward $(ifelse(allownegative, "with", "without")) negative emissions...")
            computebackward(model, G; allownegative, verbose = VERBOSE, datapath = DATAPATH, overwrite = OVERWRITE, tstop = TSTOP, cachestep = CACHESTEP)
        end
    end
end