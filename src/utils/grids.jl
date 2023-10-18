using Base.Iterators: product
using SharedArrays: SharedArray

# Utilities for regular grid
Domain = Tuple{Float32, Float32, Int};

ActionRegularGrid = NTuple{2, Vector{Float32}};
StateRegularGrid = NTuple{3, Vector{Float32}};
RegularGrid = Union{StateRegularGrid, ActionRegularGrid};
FieldGrid = Array{Float32, 3};
SharedFieldGrid = SharedArray{Float32, 3};

VectorGrid = Array{Float32, 4};
SharedVectorGrid = SharedArray{Float32, 4};

steps(grid::RegularGrid) = ntuple(i -> grid[i][2] - grid[i][1], Val(3))
Base.size(grid::RegularGrid) = prod(length, grid)
function makeregulargrid(domains::Vector{Domain})::RegularGrid
    ntuple(
        i -> collect(
            range(domains[i][1], domains[i][2]; length = domains[i][3])
        ), 
        length(domains)
    )
end

function fromgridtoarray(grid::ActionRegularGrid)::Array{Float32, 3}
    permutedims(
        collect(reinterpret(
            reshape, Float32, collect(Iterators.product(grid...))
        )),
        (2, 3, 1)
    );
end
function fromgridtoarray(grid::StateRegularGrid)::Array{Float32, 4}
    permutedims(
        collect(reinterpret(
            reshape, Float32, collect(Iterators.product(grid...))
        )),
        (2, 3, 4, 1)
    );
end