function weighting(T, m, model)
    log1p(inv(abs(μ(T, m, model.climate))))
end

function constructelasticgrid(N::NTuple{2, Int}, domains::NTuple{2, Domain{S}}, model::M) where {S, M <: UnitIAM{S}}
    Tdomain, mdomain = domains
    Tspace = range(Tdomain..., N[1])
    mspace = range(mdomain..., N[2])

    # Want high density where |μ| is small (near equilibria/tipping points)
    # Use log weighting to compress extreme values
    rawweights = [weighting(T, m, model) for T in Tspace, m in mspace]
    
    # Normalize to prevent extreme values
    weights = rawweights ./ maximum(rawweights)
    
    return ElasticGrid(N, domains, weights)
end