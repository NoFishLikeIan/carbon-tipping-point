abstract type AbstractGrid{N₁, N₂, S <: Real, R <: AbstractVector{S}} end

struct RegularGrid{N₁, N₂, S, R} <: AbstractGrid{N₁, N₂, S, R}
    domains::NTuple{2,Domain{S}}
    ranges::NTuple{2,R}

    function RegularGrid(N::NTuple{2, Int}, domains::NTuple{2, Domain{S}}) where S <: Real
        ranges = ntuple(i -> range(domains[i][1], domains[i][2]; length = N[i]), Val{2}())
        N₁, N₂ = N

        return new{N₁, N₂, S, typeof(ranges[1])}(domains, ranges)
    end
end

function Base.step(grid::RegularGrid)
    ntuple(i -> step(grid.ranges[i]), 2)
end

function inversestep(grid::RegularGrid)
    ntuple(i -> inv(step(grid.ranges[i])), 2)
end

"Returns the previous and next step of grid at positions `(i, j)`."
function steps(grid::RegularGrid{N₁, N₂, S}, i, j) where {N₁, N₂, S}
    ΔT, Δm = step(grid)
    return (ΔT, ΔT), (Δm, Δm)
end