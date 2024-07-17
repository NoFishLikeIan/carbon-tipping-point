using Revise

using DotEnv, UnPack
using CSV, DataFrames, JLD2

using DifferentialEquations
using DiffEqParamEstim, Optimization, OptimizationOptimJL

using Plots

using Model

# -- Calibration jump model that matches distribution of albedo model
begin
    DATAPATH = "data"

    const calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
    const preferences = EpsteinZin()
    const economy = Economy()
    const damages = GrowthDamages()
    const hogg = Hogg()
    const albedo = Albedo()

    albedomodel = ModelInstance(preferences, economy, damages, hogg, albedo, calibration)
end
