using UnPack

using Lux
using NNlib
using Optimisers, Zygote

include("../src/model/climate.jl")
include("../src/model/economy.jl")

include("../src/utils/derivatives.jl")
include("../src/utils/nn.jl")

