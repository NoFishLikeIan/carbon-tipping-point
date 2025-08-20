module Grid

using Base.Iterators: product, flatten

using StaticArraysCore: StaticArray
using StaticArrays

using Statistics: mean, middle
using Interpolations: interpolate, extrapolate, scale
using Interpolations: BSpline, Linear, Line
using ZigZagBoomerang: PartialQueue, dequeue!
using Graphs: SimpleGraphs

include("routines/bisection.jl")
include("grid/grid.jl")
include("grid/interpolations.jl")
include("logging.jl")

export bisection, gss, gssmin

export Point, Policy
export Domain, RegularGrid, Idx, DiagonalRedBlackQueue
export interpolateovergrid

end