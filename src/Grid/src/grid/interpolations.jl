function interpolateovergrid(V::AbstractMatrix, fromgrid::RegularGrid, togrid::RegularGrid)
    interpolateovergrid(V, fromgrid, togrid.X)
end
function interpolateovergrid(V::AbstractMatrix, fromgrid::RegularGrid, x::Point{T}) where T
    unique(interpolateovergrid(fromgrid, SMatrix{1, 1, Point{T}}(x), V))
end
function interpolateovergrid(V::AbstractMatrix, fromgrid::RegularGrid{N}, xs::AbstractMatrix{P}) where {N, T, P <: Point{T}}
    itp = extrapolate(scale(interpolate(V, BSpline(Linear())), fromgrid.ranges), Line())

    return [itp(x.T, x.m) for x ∈ xs]
end
function interpolateovergrid!(V::AbstractMatrix, fromgrid::RegularGrid{N}, xs::AbstractMatrix{P}) where {N, T, P <: Point{T}}
    itp = extrapolate(scale(interpolate(V, BSpline(Linear())), fromgrid.ranges), Line())

    return [itp(x.T, x.m) for x ∈ xs]
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