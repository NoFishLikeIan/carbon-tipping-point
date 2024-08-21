module Model

# Models
export Economy, Hogg, Albedo, Jump
export Calibration, RegionalCalibration
export Damages, GrowthDamages, LevelDamages
export equilibriumHogg
export intensity, increase, d
export μ, b, bterminal, γ, mstable, boundb, δₘ, ϕ
export criticaltemperature
export potential, density
export terminaloutputfct, outputfct
export Preferences, EpsteinZin, LogSeparable, CRRA, LogUtility
export f, g
export TippingModel, JumpModel, AbstractPlannerModel
export TippingGameModel, JumpGameModel, AbstractGameModel
export AbstractJumpModel, AbstractTippingModel
export AbstractModel
export terminalgrid
export RegionalEconomies, breakgamemodel

# Packages
using Grid: Point, Policy, Drift, RegularGrid, gss
using UnPack: @unpack
using Roots: find_zero
using FastClosures: @closure

include("models/calibration.jl")
include("models/climate.jl")
include("models/economy.jl")
include("models/preferences.jl")

struct TippingModel{D <: Damages, P <: Preferences}
    albedo::Albedo
    hogg::Hogg

    preferences::P
    damages::D

    economy::Economy
    calibration::Calibration
end

struct TippingGameModel{D <: Damages, P <: Preferences}
    albedo::Albedo
    hogg::Hogg

    preferences::NTuple{2, P}
    damages::NTuple{2, D}

    economy::NTuple{2, Economy}
    regionalcalibration::RegionalCalibration
end

struct JumpModel{D <: Damages, P <: Preferences}
    jump::Jump
    hogg::Hogg

    preferences::P
    damages::D

    economy::Economy
    calibration::Calibration
end

struct JumpGameModel{D <: Damages, P <: Preferences}
    albedo::Albedo
    jump::Jump

    preferences::NTuple{2, P}
    damages::NTuple{2, D}

    economy::NTuple{2, Economy}
    regionalcalibration::RegionalCalibration
end

AbstractPlannerModel{D, P} = Union{
    TippingModel{D, P}, JumpModel{D, P}
} where {D <: Damages, P <: Preferences}

AbstractGameModel{D, P} = Union{
    TippingGameModel{D, P}, JumpGameModel{D, P}
} where {D <: Damages, P <: Preferences}

AbstractJumpModel{D, P} = Union{
    JumpGameModel{D, P}, JumpModel{D, P}
} where {D <: Damages, P <: Preferences}

AbstractTippingModel{D, P} = Union{
    TippingGameModel{D, P}, TippingModel{D, P}
} where {D <: Damages, P <: Preferences}

AbstractModel{D, P} = Union{
    TippingModel{D, P}, JumpModel{D, P},
    TippingGameModel{D, P}, JumpGameModel{D, P}
} where {D <: Damages, P <: Preferences}

include("models/functions.jl")

end # module Model
