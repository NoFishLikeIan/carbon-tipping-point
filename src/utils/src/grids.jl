# Utilities for state grid, which have to be regular;
# -- Constructors
function makegrid(domains)
    ntuple(i -> collect(range(domains[i][1], domains[i][2]; length = domains[i][3])), length(domains))
end

function fromgridtoarray(grid)
    shifted = collect(1:length(grid)) .+ 1
    permutedims(
        reinterpret(
            reshape, Float32, Iterators.product(grid...) |> collect
        ) |> collect,
        (shifted..., 1) # Re-orders the columns
    );
end

# -- Definitions
Domain = Tuple{Float32, Float32, Int};

struct RegularGrid
    Ω::NTuple{3, Vector{Float32}}
    X::Array{Float32, 4}

    function RegularGrid(domains)
        Ω = makegrid(domains)
        X = fromgridtoarray(Ω)
        new(Ω, X)
    end
end

steps(Ω::NTuple{3, Vector{Float32}}) = ntuple(i -> Ω[i][2] - Ω[i][1], Val(3))
steps(grid::RegularGrid) = steps(grid.Ω)

dimensions(grid::RegularGrid) = length(grid.Ω)
Base.size(grid::RegularGrid) = ntuple(c -> length(grid.Ω[c]), dimensions(grid))

Base.CartesianIndices(grid::RegularGrid) = CartesianIndices(size(grid))

function isonboundary(I::CartesianIndex, grid::RegularGrid)::Bool
    any(I.I .== 1) || any(I.I .== size(grid))
end