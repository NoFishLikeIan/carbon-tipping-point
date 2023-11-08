module Model

using UnPack
using Polyester: @batch
using Statistics: mean
using Optim: optimize, minimizer, IPNewton, Newton, TwiceDifferentiable, TwiceDifferentiableConstraints, Options, only_fgh!
using Roots: Bisection, find_zero
using FastClosures: @closure
using ImageFiltering: BorderArray, Pad
using Utils: paddims

include("calibration.jl")
include("climate.jl")
include("economy.jl")

ModelInstance = Tuple{Economy, Hogg, Albedo}

include("functions.jl")
include("terminalfunctions.jl")
include("optimisation.jl")


end # module Model
