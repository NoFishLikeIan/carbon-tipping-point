module Utils

export RegularGrid, dimensions
export bisection

using Base.Iterators: product, flatten
using Polyester: @batch
using Statistics: middle

include("grids.jl")
include("derivatives.jl")
include("bisection.jl")

end # module Utils
