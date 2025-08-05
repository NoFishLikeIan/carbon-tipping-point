function interpolateovergrid(V::AbstractMatrix, fromgrid::RegularGrid, togrid::RegularGrid)
    interpolateovergrid(V, fromgrid, togrid.X)
end
function interpolateovergrid(V::AbstractMatrix, fromgrid::RegularGrid, x::Point{T}) where T
    unique(interpolateovergrid(fromgrid, SMatrix{1, 1, Point{T}}(x), V))
end
function interpolateovergrid(V::AbstractMatrix, fromgrid::RegularGrid{N}, xs::AbstractMatrix{P}) where {N, T, P <: Point{T}}
    knots = ntuple(i -> range(fromgrid.domains[i][1], fromgrid.domains[i][2], length = N), 2)
    itp = extrapolate(scale(interpolate(V, BSpline(Linear())), knots), Line())

    return [itp(x.T, x.m) for x ∈ xs]
end
function interpolateovergrid(policy::AbstractMatrix{Pol}, grid::RegularGrid{N}, xs::AbstractMatrix{P}) where {N, T, P <: Point{T}, Pol <: Policy{T}}
    knots = ntuple(i -> range(grid.domains[i][1], grid.domains[i][2], length = N), 2)
    itpχ = extrapolate(scale(interpolate(first.(policy), BSpline(Linear())), knots), Line())
    itpα = extrapolate(scale(interpolate(last.(policy), BSpline(Linear())), knots), Line())

    return [Policy(itpχ(x.T, x.m), itpα(x.T, x.m)) for x ∈ xs]
end