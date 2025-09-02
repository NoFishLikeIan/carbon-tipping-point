using Test, BenchmarkTools
using Revise
using Grid

# Test `RegularGrid`
N = (101, 11);
domains = ((0., 1.), (1., 2.));

regulargrid = RegularGrid(N, domains);

# Test IrregularGrid
function exponentialweight(x)
    μ = mean(x); σ = std(x);
    weights = @. exp(-(x - μ)^2 / 2) / 2σ
    return weights / sum(weights)
end

weights = ntuple(i -> exponentialweight(regulargrid.ranges[i]), 2)
wᵀ, wᵐ = weights

Tmin, Tmax = domains[1]
xs = range(Tmin, Tmax; length = N[1]) 
cdf = cumsum(wᵀ)
qs = Grid.inverselinearinterpolation(xs, cdf)

# Test interpolations
V = rand(size(grid)...);
densergrid = RegularGrid(2 .* N, domains);
interpolateovergrid(V, grid, densergrid)

