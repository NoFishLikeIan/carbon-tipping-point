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
function interpolatevalue(x, values::OrderedDict{S, T}) where {S, T <: Tuple}
    thresholds = values.keys
    V₀, _, G₀ = values[thresholds[1]] # Assumes that all values share timesteps and grid

    Tspace, mspace = G₀.ranges
    tspace = V₀.keys

    N₁, N₂ = size(G₀)

    H = Array{S, 4}(undef, N₁, N₂, length(tspace), length(thresholds))

    for (k, threshold) in enumerate(thresholds)
        Vₖ = first(values[threshold])
        for (i, t) in enumerate(tspace)
            H[:, :, i, k] .= Vₖ[t].H
        end
    end
    
    knots = (Tspace, mspace, tspace, thresholds)

    Hitp = interpolate(knots, H, Gridded(Linear()))

    return [Hitp(xᵢ[1], xᵢ[2], xᵢ[3], xᵢ[4]) for xᵢ in x]
end


"Chebyshev representation `Hₜ(x)`"
function chebyshevrepresentation(values::OrderedDict{S, V}, G::GR, order) where {S, N₁, N₂, V <: ValueFunction{S, N₁, N₂}, GR <: RegularGrid{N₁, N₂, S}}
    ts = values.keys
    lb = [G.domains[1][1], G.domains[2][1], ts[1]]
    ub = [G.domains[1][end], G.domains[2][end], ts[end]]

    x = chebpoints(order, lb, ub)

    Ĥ = interpolatevalue(x, values, G)

    approximatingfunction = chebinterp(Ĥ, lb, ub)

    return approximatingfunction
end

"Chebyshev representation `Hₜ(x, Tᶜ)`"
function chebyshevrepresentation(values::OrderedDict{S, T}, order) where {S, T <: Tuple}
    thresholds = values.keys
    V₀, _, G₀ = values[thresholds[1]] # Assumes that all values share timesteps and grid

    tspace = V₀.keys

    lb = [G₀.domains[1][1], G₀.domains[2][1], tspace[1], thresholds[1]]
    ub = [G₀.domains[1][end], G₀.domains[2][end], tspace[end], thresholds[end]]

    x = chebpoints(order, lb, ub)

    Ĥ = interpolatevalue(x, values)

    approximatingfunction = chebinterp(Ĥ, lb, ub)

    return approximatingfunction
end

ChebValue{S} = FastChebInterp.ChebPoly{4, S, S}

function gridevaluate!(R::RT, H::ChebValue, G::RegularGrid{N₁, N₂, S}, t, Tᶜ) where {N₁, N₂, S, RT <: AbstractMatrix{S}}
    Tspace, mspace = G.ranges

    @inbounds for i in axes(G, 1), j in axes(G, 2)
        T = Tspace[i]
        m = mspace[j]

        x = SVector{4, S}(T, m, t, Tᶜ)

        R[i, j] = H(x)
    end
end

