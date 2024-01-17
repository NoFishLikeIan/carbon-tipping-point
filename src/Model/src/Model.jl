module Model

# Models
export Economy, Hogg, Albedo, Calibration, ModelInstance
export μ, b, bterminal, γ, f, mstable

# Grid
export Domain, RegularGrid, Point, Policy, Drift, I, DiagonalRedBlackQueue
export interpolateovergrid

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

using Interpolations: interpolate, extrapolate, scale
using Interpolations: BSpline, Linear, Line
using ZigZagBoomerang: PartialQueue, dequeue!, saturate, rkey, div1, peek
using Graphs: Edge, Graph, SimpleGraphs

using JLD2: jldopen, Group

include("grid/grids.jl")
include("grid/zigzag.jl")
include("models/calibration.jl")
include("models/climate.jl")
include("models/economy.jl")


include("models/instance.jl")
include("models/functions.jl")
include("grid/interpolations.jl")

include("routines/bisection.jl")


end # module Model
