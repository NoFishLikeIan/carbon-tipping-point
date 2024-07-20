module Model

# Models
export Economy, Hogg, Albedo, Calibration, Jump
export Damages, GrowthDamages, LevelDamages
export calibrateHogg
export intensity, increase, d
export μ, b, bterminal, γ, mstable, boundb, δₘ, ϕ
export potential, density
export Preferences, EpsteinZin, LogSeparable, CRRA, LogUtility, f, g
export TippingModel, JumpModel, AbstractModel

# Packages
using Grid: Point, Policy, Drift
using UnPack: @unpack
using Roots: find_zero
using FastClosures: @closure

include("models/calibration.jl")
include("models/climate.jl")
include("models/economy.jl")
include("models/preferences.jl")

struct TippingModel{D <: Damages, P <: Preferences}
    albedo::Albedo

    preferences::P
    damages::D

    economy::Economy
    hogg::Hogg
    calibration::Calibration
end


struct JumpModel{D <: Damages, P <: Preferences}
    jump::Jump

    preferences::P
    damages::D

    economy::Economy
    hogg::Hogg
    calibration::Calibration
end

include("models/functions.jl")

end # module Model
