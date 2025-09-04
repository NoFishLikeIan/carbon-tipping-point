function Grid.interpolateovergrid(valuefunction::ValueFunction{S, N₁, N₂}, fromgrid::RegularGrid{N₁, N₂, S}, togrid::RegularGrid{N₁′, N₂′, S}) where {S, N₁, N₂, N₁′, N₂′}

    H′ = Grid.interpolateovergrid(valuefunction.H, fromgrid, togrid)
    α′ = Grid.interpolateovergrid(valuefunction.α, fromgrid, togrid)

    return ValueFunction{S, N₁′, N₂′}(H′, α′, valuefunction.t)
end