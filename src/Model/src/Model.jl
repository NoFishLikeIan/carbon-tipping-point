module Model

# Models
export Economy, Hogg, Albedo, Jump
export Calibration, RegionalCalibration
export Damages, GrowthDamages, LevelDamages
export equilibriumHogg
export intensity, increase, d
export μ, b, bterminal, costbreakdown, γ, mstable, boundb, δₘ, ϕ
export β, ε
export criticaltemperature
export potential, density
export terminaloutputfct, outputfct
export Preferences, EpsteinZin, LogSeparable, CRRA, LogUtility
export f, g, logg
export TippingModel, LinearModel, JumpModel, AbstractPlannerModel
export TippingGameModel, JumpGameModel, AbstractGameModel
export AbstractJumpModel, AbstractTippingModel
export AbstractModel
export terminalgrid
export RegionalEconomies, breakgamemodel

# Packages
using Grid: Point, Policy, RegularGrid, gss
using UnPack: @unpack
using Roots: find_zero, find_zeros
using FastClosures: @closure

include("models/calibration.jl")
include("models/climate.jl")
include("models/economy.jl")
include("models/preferences.jl")

struct LinearModel{D <: Damages, P <: Preferences}
    hogg::Hogg

    preferences::P
    damages::D

    economy::Economy
    calibration::Calibration
end

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
    TippingModel{D, P}, JumpModel{D, P}, LinearModel{D, P}
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
    AbstractPlannerModel{D, P}, AbstractGameModel{D, P}
} where {D <: Damages, P <: Preferences}

Base.broadcastable(m::AbstractModel) = Ref(m)

include("models/functions.jl")

end # module Model
