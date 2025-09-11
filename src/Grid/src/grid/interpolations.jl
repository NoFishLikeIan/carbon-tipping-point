function interpolateovergrid(V::AbstractMatrix, fromgrid::G₁, togrid::G₂) where {G₁ <: AbstractGrid, G₂ <: AbstractGrid}
    itp = linear_interpolation(fromgrid.ranges, V; extrapolation_bc = Line())

    return [itp(x.T, x.m) for x in togrid] 
end

function shrink(domain::Domain, factor)
    l, r = domain
    cut = (r - l) * factor / 2

    return (l + cut, r - cut)
end
function shrink(G::RegularGrid{N₁, N₂}, factor) where {N₁, N₂}
    RegularGrid((N₁, N₂), shrink.(G.domains, factor))
end
function halfgrid(G::RegularGrid{N₁, N₂}) where {N₁, N₂}
    RegularGrid((N₁ ÷ 2, N₂ ÷ 2), G.domains)
end