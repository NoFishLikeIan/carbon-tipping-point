# Utilities for state grid, which have to be regular;
RegularDomain = Tuple{Float32, Float32, Int};
StateGrid{N} = NTuple{N, Vector{Float32}};

steps(grid::StateGrid{N}) where N = ntuple(i -> grid[i][2] - grid[i][1], Val(N))
Base.size(grid) = prod(length, grid)

function makegrid(domains::Vector{RegularDomain})
    ntuple(
        i -> collect(
            range(domains[i][1], domains[i][2]; length = domains[i][3])
        ), 
        length(domains)
    )
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