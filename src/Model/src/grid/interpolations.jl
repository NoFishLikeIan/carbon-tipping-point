function interpolateovergrid(fromgrid::RegularGrid, V::AbstractArray, togrid::RegularGrid)
    interpolateovergrid(fromgrid, V, togrid.X)
end

function interpolateovergrid(grid::RegularGrid, V::AbstractArray{Float64, 3}, xs::AbstractArray{Point, M})::AbstractArray{Float64, M} where M
    N = size(grid, 1)
    knots = ntuple(i -> range(grid.domains[i][1], grid.domains[i][2], length = N), 3)
    itp = scale(interpolate(V, BSpline(Linear())), knots)    

    [itp(x.T, x.m, x.y) for x ∈ xs]
end

function interpolateovergrid(grid::RegularGrid, P::AbstractArray{Policy, 3}, xs::AbstractArray{Point, M})::AbstractArray{Policy, M} where M
    N = size(grid, 1)
    knots = ntuple(i -> range(grid.domains[i][1], grid.domains[i][2], length = N), 3)
    itpχ = scale(interpolate(first.(P), BSpline(Linear())), knots)
    itpα = scale(interpolate(last.(P), BSpline(Linear())), knots)

    [Policy(itpχ(x.T, x.m, x.y), itpα(x.T, x.m, x.y)) for x ∈ xs]
end