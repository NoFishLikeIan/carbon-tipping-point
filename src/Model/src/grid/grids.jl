const maxN = floor(Int, 1 / sqrt(eps(Float32)))

Domain = Tuple{Float32, Float32}

struct Point <: FieldVector{3, Float32}
    T::Float32
    m::Float32
    y::Float32
end

struct Policy <: FieldVector{2, Float32}
    χ::Float32
    α::Float32
end

struct Drift <: FieldVector{3, Float32}
    dT::Float32
    dm::Float32
    dy::Float32
end

struct TransitionProbability <: FieldVector{6, Float32}
    Δ₊T::Float32
    Δ₊m::Float32
    Δ₊y::Float32
    Δ₋T::Float32
    Δ₋m::Float32
    Δ₋y::Float32
end

PΔ₀(p::TransitionProbability) = 1 - sum(p)

struct RegularGrid{N}
    X::Array{Point, 3}
    h::Float32
    Δ::NTuple{3, Float32}
    domains::AbstractArray{Domain}

    function RegularGrid(domains::AbstractVector{Domain}, N::Int)
        if length(domains) != 3 
            throw("Domain length $(length(domains)) ≠ 3.") 
        end

        if N > maxN @warn "h < ϵ: ensure N > $maxN" end

        h = Float32(inv(N))
        Ω = (range(d[1], d[2]; length = N) for d in domains) 
        Δ = ntuple(i -> domains[i][2] - domains[i][1], 3)
        
        X = Point.(product(Ω...))

        new{N}(X, h, Δ, domains)
    end
end


Base.CartesianIndices(grid::RegularGrid) = CartesianIndices(grid.X)

emptyscalarfield(grid::RegularGrid) = Array{Float32}(undef, size(grid))
emptyvectorfield(grid::RegularGrid) = Array{Float32}(undef, size(grid)..., dimensions(grid))