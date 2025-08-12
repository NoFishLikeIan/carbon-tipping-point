const maxN = floor(Int, 1 / sqrt(eps(Float64))) # Maximum grid size
const Idx = (CartesianIndex((1, 0)), CartesianIndex((0, 1)))

Domain{T} = NTuple{2, T}

struct Point{T <: Real} <: FieldVector{3, T}
    T::T
    m::T
    M::T
end

mutable struct Policy{T <: Real} <: FieldVector{2, T} 
    χ::T
    α::T
end

struct RegularGrid{N, T <: Real}
    h::T
    X::Matrix{Point{T}}
    Δ::NTuple{2, T}
    domains::NTuple{2, Domain{T}}
end

Base.size(grid::RegularGrid) = size(grid.X)
Base.size(grid::RegularGrid, axis::Int) = size(grid.X, axis)
Base.CartesianIndices(grid::RegularGrid) = CartesianIndices(grid.X)

function Base.CartesianIndices(grid::RegularGrid, excludeboundary::Dict{Int, NTuple{2, Bool}})
    L, R = extrema(grid)

    for (dim, exclude) ∈ excludeboundary
        if exclude[1] L += Idx[dim] end
        if exclude[2] R -= Idx[dim] end
    end

    return L:R
end

function Base.extrema(::RegularGrid{N}) where N
    (CartesianIndex(1, 1), CartesianIndex(N, N))
end

function DiagonalRedBlackQueue(grid::RegularGrid; initialvector = zeros(prod(size(grid))))
    G = SimpleGraphs.grid(size(grid))
    Q = PartialQueue(G, initialvector)

    return Q
end

# Extend static array
StaticArrays.similar_type(::Type{<:Policy}, ::Type{T}, s::Size{(2,)}) where T = Policy{T}
StaticArrays.similar_type(::Type{<:Point}, ::Type{T}, s::Size{(3,)}) where T = Policy{T}

Base.similar(::Type{<:Point}, ::Type{T}) where T = Point(zero(T), zero(T), zero(T))
Base.similar(::Type{<:Policy}, ::Type{T}) where T = Policy(zero(T), zero(T))