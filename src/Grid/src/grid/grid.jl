const maxN = floor(Int, 1 / sqrt(eps(Float64))) # Maximum grid size
const Idx = (CartesianIndex((1, 0)), CartesianIndex((0, 1)))

Domain{T} = NTuple{2, T}

struct Point{T <: Real} <: FieldVector{2, T}
    T::T
    m::T
end

mutable struct Policy{T <: Real} <: FieldVector{2, T} 
    χ::T
    α::T
end

struct RegularGrid{N, T <: Real}
    h::T
    X::Array{Point{T}, 2}
    Δ::NTuple{2, T}
    domains::NTuple{2, Domain{T}}

    function RegularGrid(domains::NTuple{2, Domain{T}}, N::Int) where T
        if N > maxN @warn "h < ϵ: ensure N ≤ $maxN" end

        h = 1 / (N - 1)
        ωᵣ = map(d -> range(d[1], d[2]; length = N), domains)
        Δ = map(d -> d[2] - d[1], domains)
        
        X = Point.(Iterators.product(ωᵣ...))

        new{N, T}(h, X, Δ, domains)
    end
    function RegularGrid(domains::AbstractVector{Domain}, h::T) where T
        N = floor(Int, 1 / h) + 1
        RegularGrid(domains, N)
    end
end

Base.size(grid::RegularGrid) = size(grid.X)
Base.size(grid::RegularGrid, axis::Int) = size(grid.X, axis)
Base.CartesianIndices(grid::RegularGrid) = CartesianIndices(grid.X)

function Base.CartesianIndices(grid::RegularGrid, excludeboundary::Dict{Int, NTuple{2, Bool}})
    L, R = extrema(CartesianIndices(grid))

    for (dim, exclude) ∈ excludeboundary
        if exclude[1] L += Idx[dim] end
        if exclude[2] R -= Idx[dim] end
    end

    return L:R
end

function DiagonalRedBlackQueue(grid::RegularGrid; initialvector = zeros(prod(size(grid))))
    G = SimpleGraphs.grid(size(grid))
    Q = PartialQueue(G, initialvector)

    return Q
end