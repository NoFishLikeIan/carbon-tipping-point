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

struct RegularGrid{N}
    Ω::NTuple{N, Vector{Float32}}
    X::AbstractArray

    function RegularGrid(domains)
        Ω = makegrid(domains)
        X = fromgridtoarray(Ω)
        new{length(domains)}(Ω, X)
    end
end

steps(Ω::NTuple{N, Vector{Float32}}) where N = ntuple(i -> Ω[i][2] - Ω[i][1], Val(N))
steps(grid::RegularGrid) = steps(grid.Ω)

dimensions(grid::RegularGrid) = length(grid.Ω)
Base.size(grid::RegularGrid) = ntuple(c -> length(grid.Ω[c]), dimensions(grid))

Base.CartesianIndices(grid::RegularGrid) = CartesianIndices(size(grid))