function interpolateovergrid(fromgrid::RegularGrid, togrid::RegularGrid, V::AbstractArray)
    interpolateovergrid(fromgrid, togrid.X, V)
end

interpolateovergrid(fromgrid::RegularGrid, x::Point, V::AbstractArray) = first(interpolateovergrid(fromgrid, [x], V))
function interpolateovergrid(fromgrid::RegularGrid, xs::AbstractArray{Point, M}, V::AbstractArray) where M
    N = size(fromgrid, 1)
    knots = ntuple(i -> range(fromgrid.domains[i][1], fromgrid.domains[i][2], length = N), 2)
    itp = extrapolate(scale(interpolate(V, BSpline(Linear())), knots), Line())

    [itp(x.T, x.m) for x ∈ xs]
end

function interpolateovergrid(grid::RegularGrid, xs::AbstractArray{Point, M}, P::AbstractArray{Policy}) where M
    N = size(grid, 1)
    knots = ntuple(i -> range(grid.domains[i][1], grid.domains[i][2], length = N), 2)
    itpχ = extrapolate(scale(interpolate(first.(P), BSpline(Linear())), knots), Line())
    itpα = extrapolate(scale(interpolate(last.(P), BSpline(Linear())), knots), Line())

    [Policy(min(itpχ(x.T, x.m), 1.), min(itpα(x.T, x.m))) for x ∈ xs]
end