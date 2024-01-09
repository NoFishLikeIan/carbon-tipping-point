const maxN = floor(Int, 1 / sqrt(eps(Float64)))

const I = (
    CartesianIndex((1, 0, 0)), 
    CartesianIndex((0, 1, 0)), 
    CartesianIndex((0, 0, 1)))

Domain = Tuple{Float64, Float64}

struct Point <: FieldVector{3, Float64}
    T::Float64
    m::Float64
    y::Float64
end

struct Policy <: FieldVector{2, Float64}
    χ::Float64
    α::Float64
end

struct Drift <: FieldVector{3, Float64}
    dT::Float64
    dm::Float64
    dy::Float64
end

struct TerminalDrift <: FieldVector{2, Float64}
    dT::Float64
    dy::Float64
end

struct RegularGrid{N}
    X::Array{Point, 3}
    h::Float64
    Δ::NamedTuple{(:T, :m, :y), NTuple{3, Float64}}
    domains::AbstractArray{Domain}

    function RegularGrid(domains::AbstractVector{Domain}, N::Int)
        if length(domains) != 3 
            throw("Domain length $(length(domains)) ≠ 3.") 
        end

        if N > maxN @warn "h < ϵ: ensure N ≤ $maxN" end

        h = Float64(1 / N)
        Ω = (range(d[1], d[2]; length = N) for d in domains) 
        Δ = (;
            :T => domains[1][2] - domains[1][1],
            :m => domains[2][2] - domains[2][1],
            :y => domains[3][2] - domains[3][1]
        )
        
        X = Point.(product(Ω...))

        new{N}(X, h, Δ, domains)
    end
    function RegularGrid(domains::AbstractVector{Domain}, h::Float64)
        N = floor(Int, 1 / h)
        RegularGrid(domains, N)
    end
end

Base.size(grid::RegularGrid) = size(grid.X)
Base.size(grid::RegularGrid, axis::Int) = size(grid.X, axis)
Base.CartesianIndices(grid::RegularGrid) = CartesianIndices(grid.X)
Base.abs(d::Drift) = abs.(d)