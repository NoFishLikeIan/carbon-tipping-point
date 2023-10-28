module Model

using UnPack
using Polyester: @batch
using Optim: optimize, minimizer, IPNewton, Newton, TwiceDifferentiable, TwiceDifferentiableConstraints, Options
using Roots: Bisection, find_zero
using FastClosures: @closure
using ImageFiltering: BorderArray, Pad
using Utils: pad

include("calibration.jl")
include("climate.jl")
include("economy.jl")

ModelInstance = Tuple{Economy, Hogg, Albedo}

include("functions.jl")
include("terminal.jl")


end # module Model
