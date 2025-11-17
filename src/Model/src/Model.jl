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

export Calibration, DynamicCalibration, ConstantCalibration, DoubleExponentialCalibration
export γ, Eᵇ

export Climate, TippingClimate, LinearClimate, JumpingClimate
export Hogg, Feedback, Jump
export ExponentialDecay, ConstantDecay, SaturationRecoveryDecay
export δₘ, L, λ, μ, ∂μ∂T, ∂μ∂m
export mstable, Tstable

export Economy, Abatement, Investment, PiecewiseAbatement
export Damages
export GrowthDamages, WeitzmanGrowth, Kalkuhl, BurkeHsiangMiguel, NoDamageGrowth
export β, β′, d, D, ϕ, A, ω
export χopt, variance, consumptiongrowth

export Preferences, LogUtility, CRRA, LogSeparable, EpsteinZin
export g, f, logg, discount
export IAM, UnitIAM, determinsticIAM, linearIAM

end # module Model
