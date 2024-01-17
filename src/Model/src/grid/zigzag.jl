function DiagonalRedBlackQueue(grid::RegularGrid)
    dims = size(grid)
    G = SimpleGraphs.grid(dims)
    Q = PartialQueue(G, zeros(prod(dims)))

    return Q
end