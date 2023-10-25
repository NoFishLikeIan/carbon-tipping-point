module Model

using UnPack
using Polyester: @batch
using Optim: optimize, minimizer, Newton
using Roots: Bisection, find_zero

include("calibration.jl")
include("climate.jl")
include("economy.jl")

ModelInstance = Tuple{Economy, Hogg, Albedo}

include("functions.jl")
include("terminal.jl")


end # module Model
