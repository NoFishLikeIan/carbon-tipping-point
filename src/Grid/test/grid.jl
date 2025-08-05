using Test, BenchmarkTools
using Revise
using Grid

N = 50;
domains = ((0., 1.), (0., 1.)); 
grid = RegularGrid(domains, N);

indices = Base.CartesianIndices(grid, Dict(1 => (true, false), 2 => (false, true)));

# Indexing
@test first(indices) == CartesianIndex((2, 1))
@test last(indices) == CartesianIndex((50, 49))

V = rand(size(grid)...);
densergrid = RegularGrid(domains, 2N);
interpolateovergrid(V, grid, densergrid)

Q = DiagonalRedBlackQueue(grid)