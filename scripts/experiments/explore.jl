using Model, Grid
using JLD2

using Plots

include("../utils/saving.jl")

begin # Test model
    calibration = load_object("data/calibration.jld2")
    damages = GrowthDamages()
    hogg = Hogg()
    preferences = EpsteinZin(θ = 10., ψ = 0.75)
    economy = Economy()
    albedo = Albedo(2.5)

    model = TippingModel(albedo, hogg, preferences, damages, economy, calibration)
end

timesteps, F, policy, G = loadtotal(model; outdir = "data/simulation/constrained");
