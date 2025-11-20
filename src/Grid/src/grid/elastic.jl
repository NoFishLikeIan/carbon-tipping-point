function constructsampler(weights, domain)
    ∫w = cumsum(weights) * (domain[2] - domain[1])
    F̂ = ∫w / ∫w[end]

    N = length(weights)
    space = range(domain[1], domain[2], N)
    u = range(0, 1, N)

    elasticspace = similar(weights)
    for (j, uⱼ) in enumerate(u)
        i = searchsortedfirst(F̂, uⱼ)
        if (i == 1) || uⱼ ≤ 0
            elasticspace[j] = space[1]
        elseif uⱼ ≥ 1
            elasticspace[j] = space[end]
        else
            α = (uⱼ - F̂[i - 1]) / (F̂[i] - F̂[i - 1])
            elasticspace[j] = α * space[i] + (1 - α) * space[i - 1]
        end
    end

    return elasticspace
end

struct ElasticGrid{N₁, N₂, S, R} <: AbstractGrid{N₁, N₂, S, R}
    domains::NTuple{2,Domain{S}}
    ranges::NTuple{2,R}

    function ElasticGrid(N::NTuple{2, Int}, domains::NTuple{2, Domain{S}}, weights::NTuple{2, W}) where {S <: Real, W <: AbstractVector{S}}

        ranges = ntuple(i -> begin
            w = weights[i]
            domain = domains[i]

            if length(w) ≠ N[i]
                throw(ArgumentError("Length of weights[$(i)] ($(length(w))) must equal N[$(i)] ($(N[i]))"))
            end

            return constructsampler(w, domain)
        end, 2)

        for (i, r) in enumerate(ranges)
            if any(diff(r) .≤ 0)
                throw(ArgumentError("Grid points in dimension $(i) must be strictly increasing"))
            end
        end

        return new{N[1],N[2],S,typeof(ranges[1])}(domains, ranges)
    end
    function ElasticGrid(N, domains, weights::W) where {S <: Real, W <: AbstractMatrix{S}}
        wᵀ = dropdims(sum(weights, dims = 2); dims = 2)
        wᵐ = dropdims(sum(weights, dims = 1); dims = 1)
        return ElasticGrid(N, domains, (wᵀ, wᵐ))
    end
end

"Returns the previous and next step of grid at positions `(i, j)`. Returns `Inf` if step is on edge."
function steps(grid::ElasticGrid{N₁, N₂, S}, i, j) where {N₁, N₂, S}
    Tspace, mspace = grid.ranges

    ΔT₋ = i == 1 ? typemax(S) : Tspace[i] - Tspace[i - 1]
    ΔT₊ = i == N₁ ? typemax(S) : Tspace[i + 1] - Tspace[i]
    
    Δm₋ = j == 1 ? typemax(S) : mspace[j] - mspace[j - 1]
    Δm₊ = j == N₂ ? typemax(S) : mspace[j + 1] - mspace[j]

    return (ΔT₋, ΔT₊), (Δm₋, Δm₊)
end