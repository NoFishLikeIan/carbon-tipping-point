struct ElasticGrid{N₁, N₂, S <: Real, R <: Vector{S}}
    domains::NTuple{2,Domain{S}}
    ranges::NTuple{2,R}
    X::Matrix{Point{S}}

    function ElasticGrid(N::NTuple{2, Int}, domains::NTuple{2, Domain{S}}, weights::NTuple{2, Vector{S}}) where S <: Real
        wᵀ, wᵐ = weights
        
        X = [Point(T, m) for T in ranges[1], m in ranges[2]]

        return new{N[1],N[2],S,typeof(ranges[1])}(domains, ranges, X)
    end
end