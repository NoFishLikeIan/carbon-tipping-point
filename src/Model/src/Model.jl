module Model

# Models
export Economy, Hogg, Albedo, Calibration, ModelInstance

# Grid
export Domain, RegularGrid, Point, Policy, Drift
export dimensions, emptyscalarfield, emptyvectorfield

# Functions
export hjb, objective!, optimalpolicy, policyovergrid!
export hjbterminal, terminalfoc, terminalpolicyovergrid!, optimalterminalpolicy

# Routines
export bisection

# Packages
using Base.Iterators: product, flatten
using Statistics: mean, middle
using LinearAlgebra: dot
using UnPack: @unpack
using Polyester: @batch
using FastClosures: @closure

using StaticArraysCore: StaticArray
using StaticArrays: FieldVector

include("models/calibration.jl")
include("models/climate.jl")
include("models/economy.jl")

include("grid/grids.jl")
include("grid/approximate.jl")

include("models/instance.jl")

include("models/functions.jl")
include("models/terminalfunctions.jl")

# include("routines/optimisation.jl")
include("routines/bisection.jl")


end # module Model
