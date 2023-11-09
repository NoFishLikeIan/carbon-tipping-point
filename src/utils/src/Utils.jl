module Utils

export RegularGrid, steps, dimensions, paddims
export central∇, central∇!, central∂, central∂!, dir∇, dir∇!, dir∂, dir∂!, ∂², ∂²!

using Base.Iterators: product
using ImageFiltering: BorderArray, Pad
using Polyester: @batch

include("grids.jl")
include("derivatives.jl")

end # module Utils
