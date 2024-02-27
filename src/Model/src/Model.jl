module Model

# Models
export Economy, Hogg, Albedo, Calibration, Jump
export calibrateHogg
export intensity, increase
export μ, b, bterminal, γ, mstable, boundb, δₘ
export potential, density
export Preferences, EpsteinZin, LogSeparable, CRRA, LogUtility, f
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
    hogg::Hogg = Hogg()
    albedo::Albedo = Albedo()
    calibration::Calibration
end

@kwdef struct ModelBenchmark
    preferences::Preferences = EpsteinZin()
    economy::Economy = Economy()
    hogg::Hogg = Hogg()
    jump::Jump = Jump()
    calibration::Calibration
end



include("models/functions.jl")

end # module Model
