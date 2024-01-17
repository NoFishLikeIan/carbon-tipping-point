module Grid

export bisection, gss

export Point, Policy, Drift
export Domain, RegularGrid, I, DiagonalRedBlackQueue
export interpolateovergrid

using Base.Iterators: product, flatten

using StaticArraysCore: StaticArray
using StaticArrays: FieldVector

using Statistics: mean, middle
using Interpolations: interpolate, extrapolate, scale
using Interpolations: BSpline, Linear, Line
using ZigZagBoomerang: PartialQueue, dequeue!, saturate, rkey, div1, peek
using Graphs: Edge, Graph, SimpleGraphs

include("routines/bisection.jl")
include("grid/grids.jl")
include("grid/interpolations.jl")

end