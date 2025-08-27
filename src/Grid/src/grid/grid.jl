const maxN = floor(Int, 1 / sqrt(eps(Float64))) # Maximum grid size

Domain{T} = NTuple{2,T}

struct Point{T<:Real} <: FieldVector{2,T}
    T::T
    m::T
end

# Extend static array
StaticArrays.similar_type(::Type{<:Point}, ::Type{T}, s::Size{(2,)}) where T = Point{T}
Base.similar(::Type{<:Point}, ::Type{T}) where T = Point(zero(T), zero(T))

struct RegularGrid{N₁, N₂, S <: Real, R <: StepRangeLen{S}}
    domains::NTuple{2,Domain{S}}
    ranges::NTuple{2,R}
    X::Matrix{Point{S}}

    function RegularGrid(N::NTuple{2, Int}, domains::NTuple{2, Domain{S}}) where S <: Real
        ranges = ntuple(i -> range(domains[i][1], domains[i][2]; length = N[i]), Val{2}())
        X = [Point(T, m) for T in ranges[1], m in ranges[2]]

        return new{N[1],N[2],S,typeof(ranges[1])}(domains, ranges, X)
    end
end

Base.size(grid::RegularGrid) = size(grid.X)
Base.size(grid::RegularGrid, d::Int) = size(grid.X, d)
Base.axes(grid::RegularGrid, d) = axes(grid.X, d)
Base.axes(grid::RegularGrid) = axes(grid.X)
Base.length(grid::RegularGrid) = length(grid.X)

function Base.extrema(grid::RegularGrid)
    ntuple(i -> grid.domains[i][2] - grid.domains[i][1], 2)
end

function Base.step(grid::RegularGrid)
    ntuple(i -> step(grid.ranges[i]), 2)
end

Base.LinearIndices(grid::RegularGrid) = LinearIndices(grid.X)
Base.CartesianIndices(grid::RegularGrid) = CartesianIndices(grid.X)

function LinearIndex(idx::CartesianIndex, G::RegularGrid)
    LinearIndex(idx.I, G)
end

function LinearIndex((i, j)::NTuple{2, Int}, G::RegularGrid{N₁}) where N₁
    i + (j - 1) * N₁
end

function Base.CartesianIndex(k::Int, ::RegularGrid{N₁}) where {N₁}
    i = ((k - 1) % N₁) + 1
    j = ((k - 1) ÷ N₁) + 1
    return CartesianIndex(i, j)
end

function closestgridpoint(Xᵢ::Point, grid::RegularGrid)
    Tspace, mspace = grid.ranges
    i = argmin(i -> abs(Tspace[i] - Xᵢ.T), axes(grid, 1))
    j = argmin(j -> abs(mspace[j] - Xᵢ.m), axes(grid, 2))
    
    return i, j
end