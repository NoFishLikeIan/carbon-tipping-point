module Utils

export RegularGrid, steps, dimensions, paddims, bisection
export central∇, central∇!, central∂, central∂!, dir∇, dir∇!, dir∂, dir∂!, ∂², ∂²!

using Base.Iterators: product
using ImageFiltering: BorderArray, Pad
using Polyester: @batch
using Statistics: middle

include("grids.jl")
include("derivatives.jl")
include("bisection.jl")

end # module Utils
