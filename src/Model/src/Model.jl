module Model

using Roots: find_zero
using UnPack: @unpack
using LogExpFunctions: logistic

include("models/calibration.jl")
include("models/climate.jl")
include("models/economy.jl")
include("models/preferences.jl")
include("models/models.jl")
include("models/functions.jl")

export Calibration, RegionalCalibration
export γ, Eᵇ

export Feedback, Jump, Hogg
export δₘ, L, λ, μ, Tstable, intensity

export Damages, GrowthDamages, LevelDamages, Economy, RegionalEconomies
export β, β′, d, D, ϕ, A

export AbstractModel, LinearModel, TippingModel, JumpModel
export Preferences, LogUtility, CRRA, LogSeparable, EpsteinZin
export b, bterminal, outputfct, terminaloutputfct


end # module Model
