using Test, BenchmarkTools
using Revise
using Grid

# Test `RegularGrid`
N = (101, 100);
domains = ((0., 2.), (1., 2.));

regulargrid = RegularGrid(N, domains);

# Test elastic grid
μ(T) = T - T^3 / 3 
w(T) = abs(1 - T^2)
domain = domains[1]
weights = [w(T) for T in range(domain..., N[1])]

uniformweights = ones(N[2])

elasticgrid = ElasticGrid(N, domains, (weights, uniformweights))

# Test interpolations
V = rand(size(grid)...);
densergrid = RegularGrid(2 .* N, domains);
interpolateovergrid(V, grid, densergrid)

