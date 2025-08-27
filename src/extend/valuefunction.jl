function Grid.interpolateovergrid(valuefunction::ValueFunction{S, N₁, N₂}, fromgrid::RegularGrid{N₁, N₂, S}, togrid::RegularGrid) where {S, N₁, N₂}

    H′ = Grid.interpolateovergrid(valuefunction.H, fromgrid, togrid)
    α′ = Grid.interpolateovergrid(valuefunction.α, fromgrid, togrid)

    return ValueFunction{S, N₁, N₂}(H′, α′, valuefunction.t)
end