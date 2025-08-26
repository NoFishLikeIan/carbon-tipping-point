using Test, BenchmarkTools
using Revise
using Grid

N = (50, 51);
domains = ((0., 1.), (1., 2.)); 
grid = RegularGrid(N, domains);

V = rand(size(grid)...);
densergrid = RegularGrid(2 .* N, domains);
interpolateovergrid(V, grid, densergrid)