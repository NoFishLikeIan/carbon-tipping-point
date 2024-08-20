include("utils/saving.jl")
include("markov/terminal.jl")
include("markov/game.jl")

env = DotEnv.config(".envgame")
DATAPATH = get(env, "DATAPATH", "data")
SIMPATH = get(env, "SIMULATIONPATH", "game")

datapath = joinpath(DATAPATH, SIMPATH)

N = getnumber(env, "N", 31; type = Int)
M = getnumber(env, "M", 10; type = Int)

VERBOSE = getbool(env, "VERBOSE", false)
RUNTERMINAL = getbool(env, "RUNTERMINAL", false)
RUNBACKWARDS = getbool(env, "RUNBACKWARDS", false)
OVERWRITE = getbool(env, "OVERWRITE", false)
TOL = getnumber(env, "TOL", 1e-3)
TSTOP = getnumber(env, "TSTOP", 0.)
CACHESTEP = getnumber(env, "CACHESTEP", 1 / 4)

OVERWRITE && @warn "Running in overwrite mode!"

# Construct model
begin
    thresholds = [2.5];
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
    
    if RUNTERMINAL
        highmodel, lowmodel = breakgamemodel(model)
        for (label, regionalmodel) in zip(("high", "low"), breakgamemodel(model))
            VERBOSE && println("Solving for $label income countries...")

            computeterminal(regionalmodel, G; verbose = VERBOSE, datapath, alternate = true, tol = TOL, overwrite = OVERWRITE, addpath = label) 
        end
    end

    if RUNBACKWARDS
        VERBOSE && println("Running backward game simulation...")
        computebackward(model, G; verbose = VERBOSE, datapath, overwrite = OVERWRITE, tstop = TSTOP, cachestep = CACHESTEP)
    end
end