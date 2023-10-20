using Base.Iterators: product

# Utilities for state grid, which have to be regular;
RegularDomain = Tuple{Float32, Float32, Int};
StateGrid = NTuple{3, Vector{Float32}};

FieldGrid = Array{Float32, 3};
VectorGrid = Array{Float32, 4};
SVectorGrid = SharedArray{Float32, 4};

steps(grid) = ntuple(i -> grid[i][2] - grid[i][1], Val(3))
Base.size(grid) = prod(length, grid)

function makegrid(domains::Vector{RegularDomain})
    ntuple(
        i -> collect(
            range(domains[i][1], domains[i][2]; length = domains[i][3])
        ), 
        length(domains)
    )
end

function fromgridtoarray(grid)::VectorGrid
    permutedims(
        collect(reinterpret(
            reshape, Float32, collect(Iterators.product(grid...))
        )),
        (2, 3, 4, 1) # Re-orders the columns
    );
end

# Utilities for action grid, relying on LatinHypercubeSampling
LatinDomain = Tuple{Float32, Float32}

function makegrid(domains::Vector{LatinDomain}, n)
    Float32.(scaleLHC(randomLHC(n, 2), domains))
end