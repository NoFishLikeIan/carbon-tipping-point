using PGFPlotsX
using Plots
using Roots
using LaTeXStrings
using DifferentialEquations
using DotEnv, JLD2

using Colors, ColorSchemes

PALETTE = colorschemes[:grays];

includet("utils.jl")
includet("../utils/saving.jl")

begin # Environment variables
    env = DotEnv.config(".env.game")
    plotpath = get(env, "PLOTPATH", "plots")
    datapath = get(env, "DATAPATH", "data")
    simulationpath = get(env, "SIMULATIONPATH", "simulations")

    BASELINE_YEAR = 2020
    SAVEFIG = false

    calibration = load_object(joinpath(datapath, "calibration.jld2"))
    regionalcalibration = load_object(joinpath(datapath, "regionalcalibration.jld2"))
end;

begin # Load simulations
    threshold = 1.5

    # -- Climate
    hogg = Hogg()
    tipping = Albedo(threshold)

    # -- Economy and Preferences
    preferences = EpsteinZin();
    oecdeconomy, roweconomy = RegionalEconomies()

    oecdmodel = TippingModel(albedo, hogg, preferences, LevelDamages(), oecdeconomy)
    rowmodel = TippingModel(albedo, hogg, preferences, GrowthDamages(), roweconomy)
    models = [oecdmodel, rowmodel]
end

timesteps, F, policy, G, models = loadgame(models; outdir = simulationpath)