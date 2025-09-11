function Grid.interpolateovergrid(valuefunction::ValueFunction{S, N₁, N₂}, fromgrid::G₁, togrid::G₂) where {N₁, N₂, M₁, M₂, S, G₁ <: AbstractGrid{N₁, N₂, S}, G₂ <: AbstractGrid{M₁, M₂, S}}

    H′ = Grid.interpolateovergrid(valuefunction.H, fromgrid, togrid)
    α′ = Grid.interpolateovergrid(valuefunction.α, fromgrid, togrid)

    return ValueFunction{S, M₁, M₂}(H′, α′, valuefunction.t)
end