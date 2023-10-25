module Model

using UnPack
using Polyester: @batch
using Optim: optimize, minimizer, Newton
using Roots: Bisection, find_zero

include("climate.jl")
include("economy.jl")

ModelInstance = Union{Economy, Hogg, Albedo}

include("calibration.jl")

include("pdes.jl")
include("terminalpdes.jl")


end # module Model
