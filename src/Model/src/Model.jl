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

export Calibration, RegionalCalibration
export γ, Eᵇ

export Feedback, Jump, Hogg
export δₘ, L, λ, μ, ∂μ∂T
export mstable, Tstable

export Damages
export GrowthDamages, WeitzmanGrowth, Kalkuhl, NoDamageGrowth
export LevelDamages, WeitzmanLevel, NoDamageLevel
export Economy, RegionalEconomies
export β, β′, d, D, ϕ, A, ω
export Φ, Bterminal, χopt

export g, f, logg, discount

export AbstractModel, LinearModel, TippingModel, JumpModel
export Preferences, LogUtility, CRRA, LogSeparable, EpsteinZin

end # module Model
