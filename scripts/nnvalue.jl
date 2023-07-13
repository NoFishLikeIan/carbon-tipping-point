using UnPack

using Flux

include("../src/model/climate.jl")
include("../src/model/economic.jl")

include("../src/loss/foc.jl")
include("../src/loss/loss.jl")