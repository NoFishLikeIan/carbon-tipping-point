using DotEnv

using UnPack
using Flux

include("../src/model/climate.jl")
include("../src/model/economy.jl")

include("../src/loss/foc.jl")
include("../src/loss/loss.jl")

env = DotEnv.config()
DATAPATH = get(env, "DATAPATH", "data")

begin # Load calibrated data
    @load joinpath(DATAPATH, "calibration.jld2") ipcc
    @unpack Eᵇ, Tᵇ, Mᵇ, N₀, γparameters = ipcc

    γᵇ(t) = γ(t, γparameters[1:3], γparameters[4])
end