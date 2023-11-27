module Model

# Models
export Economy, Hogg, Albedo, Calibration, ModelInstance

# Grid
export Domain, RegularGrid
export dimensions

# Functions
export hjb, objective!, optimalpolicy, policyovergrid!
export hjbterminal, terminalfoc, terminalpolicyovergrid!, optimalterminalpolicy

# Routines
export bisection

# Packages
using UnPack
using Polyester: @batch
using Statistics: mean, middle
using Optim: optimize, minimizer, IPNewton, Newton, TwiceDifferentiable, TwiceDifferentiableConstraints, Options, only_fgh!
using FastClosures: @closure
using Base.Iterators: product, flatten



include("models/calibration.jl")
include("models/climate.jl")
include("models/economy.jl")

ModelInstance = Tuple{Economy, Hogg, Albedo}

include("grid/grids.jl")

include("models/functions.jl")
include("models/terminalfunctions.jl")

include("routines/optimisation.jl")
include("routines/bisection.jl")


end # module Model
