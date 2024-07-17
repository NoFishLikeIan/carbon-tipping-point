module Model

# Models
export Economy, Hogg, Albedo, Calibration, Jump
export Damages, GrowthDamages, LevelDamages
export calibrateHogg
export intensity, increase, d
export μ, b, bterminal, γ, mstable, boundb, δₘ, ϕ
export potential, density
export Preferences, EpsteinZin, LogSeparable, CRRA, LogUtility, f, g
export ModelInstance, ModelBenchmark

# Packages
using Grid: Point, Policy, Drift
using UnPack: @unpack
using Roots: find_zero
using FastClosures: @closure

include("models/calibration.jl")
include("models/climate.jl")
include("models/economy.jl")
include("models/preferences.jl")

@kwdef struct ModelInstance
    preferences::Preferences = EpsteinZin()
    economy::Economy = Economy()
    damages::Damages = GrowthDamages()
    hogg::Hogg = Hogg()
    albedo::Albedo = Albedo()
    calibration::Calibration
end

@kwdef struct ModelBenchmark
    preferences::Preferences = EpsteinZin()
    economy::Economy = Economy()
    damages::Damages = GrowthDamages()
    hogg::Hogg = Hogg()
    jump::Jump = Jump()
    calibration::Calibration
end

include("models/functions.jl")

end # module Model
