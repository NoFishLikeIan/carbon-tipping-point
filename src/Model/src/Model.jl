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
export TippingModel, LinearModel, JumpModel
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

struct JumpModel{D <: Damages, P <: Preferences}
    jump::Jump
    hogg::Hogg

    preferences::P
    damages::D

    economy::Economy
    calibration::Calibration
end

AbstractModel{D, P} = Union{
    TippingModel{D, P}, JumpModel{D, P}, LinearModel{D, P}
} where {D <: Damages, P <: Preferences}

Base.broadcastable(m::AbstractModel) = Ref(m)

include("models/functions.jl")

end # module Model
