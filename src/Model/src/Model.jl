module Model

using Roots: find_zero, find_zeros
using UnPack: @unpack
using LogExpFunctions: logistic
using FastPow: @fastpow

include("models/calibration.jl")
include("models/climate.jl")
include("models/economy.jl")
include("models/preferences.jl")
include("models/models.jl")
include("logging.jl")

export Calibration, DynamicCalibration, ConstantCalibration, RegionalCalibration
export γ, Eᵇ

export Climate, TippingClimate, LinearClimate, JumpingClimate
export Hogg, Feedback, Jump
export ExponentialDecay, ConstantDecay
export δₘ, L, λ, μ, ∂μ∂T, ∂μ∂m
export mstable, Tstable

export Economy, Abatement, Investment
export Damages
export GrowthDamages, WeitzmanGrowth, Kalkuhl, NoDamageGrowth
export LevelDamages, WeitzmanLevel, NoDamageLevel
export β, β′, d, D, ϕ, A, ω
export χopt

export Preferences, LogUtility, CRRA, LogSeparable, EpsteinZin
export g, f, logg, discount
export IAM, UnitIAM

end # module Model
