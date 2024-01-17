using Test: @test
using Revise
using Grid

N = 3;
domains = [(0., 1.), (0., 1.), (0., 1.)]; 
grid = RegularGrid(domains, N);

indices = Base.CartesianIndices(grid, Dict(1 => (true, false), 3 => (false, true)));

# Indexing
@test first(indices) == CartesianIndex((2, 1, 1))
@test last(indices) == CartesianIndex((3, 3, 2))

V = rand(size(grid)...);
densergrid = RegularGrid(domains, 2N);
interpolateovergrid(grid, V, densergrid);