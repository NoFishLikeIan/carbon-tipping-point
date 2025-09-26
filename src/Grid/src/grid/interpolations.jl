function interpolateovergrid(V::AbstractMatrix, fromgrid::G₁, togrid::G₂) where {G₁ <: AbstractGrid, G₂ <: AbstractGrid}
    itp = linear_interpolation(fromgrid.ranges, V; extrapolation_bc = Line())

    Tspace, mspace = togrid.ranges

    return [itp(T, m) for T in Tspace, m in mspace] 
end

function shrink(domain::Domain, factors)
    leftfactor, rightfactor = factors
    l, r = domain
    
    leftcut = (r - l) * leftfactor
    rightcut = (r - l) * rightfactor

    return (l + leftcut, r - rightcut)
end
function shrink(G::RegularGrid{N₁, N₂}, factor) where {N₁, N₂}
    RegularGrid((N₁, N₂), shrink.(G.domains, Ref(factor)))
end
function halfgrid(G::RegularGrid{N₁, N₂}) where {N₁, N₂}
    RegularGrid((N₁ ÷ 2, N₂ ÷ 2), G.domains)
end