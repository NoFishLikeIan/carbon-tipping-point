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

function shrink(domain::Domain, factor)
    l, r = domain
    cut = (r - l) * (1 - factor) / 2

    return (l + cut, r - cut)
end

function shrink(G::RegularGrid, factor)
    RegularGrid(N, shrink.(G.domains, factor))
end

function halfgrid(G::RegularGrid)
    RegularGrid(G.N .÷ 2, G.domains)
end