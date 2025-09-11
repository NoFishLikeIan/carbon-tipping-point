function weighting(T, m, model; contrast = 1 / 4)
    dT = μ(T, m, model.climate)
    baseweight = exp(-contrast * abs(dT))
    return baseweight
end

function constructelasticgrid(N::NTuple{2, Int}, domains::NTuple{2, Domain{S}}, model::M; weightkwargs...) where {S, M <: UnitIAM{S}}
    Tdomain, mdomain = domains
    Tspace = range(Tdomain..., N[1])
    mspace = range(mdomain..., N[2])
    
    rawweights = [weighting(T, m, model; weightkwargs...) for T in Tspace, m in mspace]
    weights = rawweights ./ maximum(rawweights)
    
    return ElasticGrid(N, domains, weights)
end

function Grid.halfgrid(G::ElasticGrid{N₁, N₂, S}, model; weightkwargs...) where {N₁, N₂, S}
    Tdomain, mdomain = G.domains
    N₁′ = N₁ ÷ 2
    N₂′ = N₂ ÷ 2
    Tspace = range(Tdomain..., N₁′)
    mspace = range(mdomain..., N₂′)
    
    rawweights = [weighting(T, m, model; weightkwargs...) for T in Tspace, m in mspace]
    weights = rawweights ./ maximum(rawweights)

    return ElasticGrid((N₁′, N₂′), G.domains, weights)
end
function Grid.shrink(G::ElasticGrid{N₁, N₂, S}, factor, model; weightkwargs...) where {N₁, N₂, S}
    newdomains = Grid.shrink.(G.domains, factor)
    Tspace = range(newdomains[1]..., N₁)
    mspace = range(newdomains[2]..., N₂)
    
    rawweights = [weighting(T, m, model; weightkwargs...) for T in Tspace, m in mspace]
    weights = rawweights ./ maximum(rawweights)

    return ElasticGrid((N₁, N₂), newdomains, weights)
end

function Grid.halfgrid(G::RegularGrid, model)
    return halfgrid(G)
end
function Grid.shrink(G::RegularGrid, factor, model)
    return shrink(G, factor)
end