module Grid

export bisection, gss, gssmin

export Point, Policy
export Domain, RegularGrid, I, DiagonalRedBlackQueue
export interpolateovergrid

using Base.Iterators: product, flatten

using StaticArraysCore: StaticArray
using StaticArrays: FieldVector, MVector

using Statistics: mean, middle
using Interpolations: interpolate, extrapolate, scale
using Interpolations: BSpline, Linear, Line
using ZigZagBoomerang: PartialQueue, dequeue!
using Graphs: SimpleGraphs

include("routines/bisection.jl")
include("grid/grids.jl")
include("grid/interpolations.jl")

end