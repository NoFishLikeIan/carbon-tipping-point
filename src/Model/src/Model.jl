module Model

# Models
export Economy, Hogg, Albedo, Calibration, ModelInstance
export μ, b, bterminal, γ, f

# Grid
export Domain, RegularGrid, Point, Policy, Drift
export dimensions, emptyscalarfield, emptyvectorfield
export I

# Functions
export driftbounds, driftbounds!
export optimalpolicy, optimalterminalpolicy

# Routines
export bisection, gss
export backwardsimulation!, terminaljacobi!

# Packages
using Base.Iterators: product, flatten
using Statistics: mean, middle
using LinearAlgebra: dot
using UnPack: @unpack
using Polyester: @batch
using FastClosures: @closure

using StaticArraysCore: StaticArray
using StaticArrays: FieldVector
using DataStructures: PriorityQueue, dequeue_pair!

using Optim: TwiceDifferentiableConstraints, TwiceDifferentiable, only_fgh!, optimize, IPNewton, minimizer, GoldenSection, Options

using JLD2: jldopen, Group

include("grid/grids.jl")
include("models/calibration.jl")
include("models/climate.jl")
include("models/economy.jl")


include("models/instance.jl")

include("models/functions.jl")

include("routines/optimisation.jl")
include("routines/bisection.jl")
include("routines/jacobi.jl")


end # module Model
