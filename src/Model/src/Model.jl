module Model

# Models
export Economy, Hogg, Albedo, Calibration
export μ, b, bterminal, γ, mstable, boundb, δₘ
export potential, density
export Preferences, EpsteinZin, LogSeparable, CRRA, LogUtility, f
export ModelInstance

# Packages
using Grid: Point, Policy, Drift
using UnPack: @unpack

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

include("models/functions.jl")

end # module Model
