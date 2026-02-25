function interpolatevalue(x, values::OrderedDict{S, V}, G::GR) where {S, N₁, N₂, V <: ValueFunction{S, N₁, N₂}, GR <: RegularGrid{N₁, N₂, S}}
    Tspace, mspace = G.ranges
    tspace = values.keys

    H = Array{S, 3}(undef, N₁, N₂, length(tspace))

    for (i, value) in enumerate(values.vals)
        H[:, :, i] .= value.H
    end
    
    knots = (Tspace, mspace, tspace)

    Hitp = interpolate(knots, H, Gridded(Linear()))

    return [Hitp(xᵢ[1], xᵢ[2], xᵢ[3]) for xᵢ in x]
end

"Chebyshev representation `Hₜ(x)` from finite-difference representation `OrderedDict{<:Real, <: ValueFunction}`"
function chebyshevrepresentation(values::OrderedDict{S, V}, G::GR; order = (100,100, 100)) where {S, N₁, N₂, V <: ValueFunction{S, N₁, N₂}, GR <: RegularGrid{N₁, N₂, S}}
    ts = values.keys
    lb = [G.domains[1][1], G.domains[2][1], ts[1]]
    ub = [G.domains[1][end], G.domains[2][end], ts[end]]

    x = chebpoints(order, lb, ub)

    Ĥ = interpolatevalue(x, values, G)

    approximatingfunction = chebinterp(Ĥ, lb, ub)

    return approximatingfunction
end