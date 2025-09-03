using Test, BenchmarkTools
using Revise
using Grid

# Test `RegularGrid`
N = (101, 100);
domains = ((0., 1.), (1., 2.));

regulargrid = RegularGrid(N, domains);

# Test IrregularGrid


# Test interpolations
V = rand(size(grid)...);
densergrid = RegularGrid(2 .* N, domains);
interpolateovergrid(V, grid, densergrid)

