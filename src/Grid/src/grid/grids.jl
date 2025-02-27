const maxN = floor(Int, 1 / sqrt(eps(Float64)))

const I = (
    CartesianIndex((1, 0)), 
    CartesianIndex((0, 1))
)

Domain = Tuple{Float64, Float64}

struct Point <: FieldVector{2, Float64}
    T::Float64
    m::Float64
end

mutable struct Policy{T <: Real} <: FieldVector{2, T} 
    χ::T
    α::T
end

struct RegularGrid{N}
    X::Array{Point, 2}
    h::Float64
    Δ::NamedTuple{(:T, :m), NTuple{2, Float64}}
    domains::AbstractArray{Domain}

    function RegularGrid(domains::AbstractVector{Domain}, N::Int)
        if length(domains) != 2 
            throw("Domain length $(length(domains)) ≠ 2.") 
        end

        if N > maxN @warn "h < ϵ: ensure N ≤ $maxN" end

        h = 1 / (N - 1)
        ωᵣ = (range(d[1], d[2]; length = N) for d in domains) 
        Δ = (;
            :T => domains[1][2] - domains[1][1],
            :m => domains[2][2] - domains[2][1]
        )
        
        X = Point.(product(ωᵣ...))

        new{N}(X, h, Δ, domains)
    end
    function RegularGrid(domains::AbstractVector{Domain}, h::Float64)
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
        if exclude[1] L += I[dim] end
        if exclude[2] R -= I[dim] end
    end

    return L:R
end

function DiagonalRedBlackQueue(grid::RegularGrid; initialvector = zeros(prod(size(grid))))
    G = SimpleGraphs.grid(size(grid))
    Q = PartialQueue(G, initialvector)

    return Q
end