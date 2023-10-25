module Model

using UnPack
using Polyester: @batch
using Optim: optimize, minimizer, IPNewton, Newton, TwiceDifferentiable, TwiceDifferentiableConstraints, Options
using Roots: Bisection, find_zero
using FastClosures: @closure

include("calibration.jl")
include("climate.jl")
include("economy.jl")

ModelInstance = Tuple{Economy, Hogg, Albedo}

include("functions.jl")
include("terminal.jl")


end # module Model
