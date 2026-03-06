struct SimplexPolicies{K, S, TS <: StaticVector{K, S}, PS <: AbstractArray{S, 4}, GR <: RegularGrid, T <: AbstractVector}
    policiesarray::PS
    thresholds::TS
    G::GR
    ts::T
end
function SimplexPolicies(pathdict::OrderedDict; tspan = (0, Inf))
    K = length(pathdict)
    policyarray, G, ts, thresholds = loadpolicyarray(pathdict; tspan)
    staticthresholds = SVector{K}(thresholds...)
    return SimplexPolicies(policyarray, staticthresholds, G, ts)
end

"Load policies in `paths` into an array"
function loadpolicyarray(paths::PS; tspan = (0., Inf)) where {S, PS <: OrderedDict{S, String}}
    thresholds = paths.keys
    firstpath = first(paths.vals)
    v, _, G = loadtotal(firstpath; tspan)
    ts = collect(v.keys)

    # Assume ts and grids to be constant
    policyarray = Array{S}(undef, size(G, 1), size(G, 2), length(ts), length(thresholds))
    for (kdx, threshold) in enumerate(thresholds)
        thresholdpath = paths[threshold]
        v = first(loadtotal(thresholdpath; tspan))

        for (tdx, t) in enumerate(ts)
            vₜ = v[t]
            policyarray[:, :, tdx, kdx] .= vₜ.α
        end
    end

    return policyarray, G, ts, thresholds
end

function Base.show(io::IO, sp::SimplexPolicies{K, S}) where {K, S}
    print(io, "SimplexPolicies{K = $K | Tᶜ ∈ {$(join(sp.thresholds, ", "))} °C}")
end
function Base.show(io::IO, ::MIME"text/plain", sp::SimplexPolicies)
    show(io, sp)
end

function weightedpolicy(x::Point{S}, t::S, weights::W, policybasis::P) where {K, S, W <: StaticVector{K}, P <: SimplexPolicies{K, S}}
    @unpack policiesarray, G, ts = policybasis
    Tspace, mspace = G.ranges

    N₁, N₂ = size(G)
    M = length(ts)

    T₀, T₁ = first(Tspace), last(Tspace)
    m₀, m₁ = first(mspace), last(mspace)
    t₀, t₁ = first(ts), last(ts)

    dT⁻¹ = inv(T₁ - T₀)
    dm⁻¹ = inv(m₁ - m₀)
    dt⁻¹ = inv(t₁ - t₀)

    # Normalize in unit space (allow values outside [0, 1] for linear extrapolation)
    Tₙ = (x.T - T₀) * dT⁻¹
    mₙ = (x.m - m₀) * dm⁻¹
    tₙ = (t - t₀) * dt⁻¹

    # Convert unit coordinates to cell coordinates
    Tₛ = Tₙ * (N₁ - 1)
    mₛ = mₙ * (N₂ - 1)
    tₛ = tₙ * (M - 1)

    i = clamp(floor(Int, Tₛ) + 1, 1, N₁ - 1)
    j = clamp(floor(Int, mₛ) + 1, 1, N₂ - 1)
    s = clamp(floor(Int, tₛ) + 1, 1, M - 1)

    # Interpolation weights
    ξ = Tₛ - (i - 1)
    η = mₛ - (j - 1)
    ζ = tₛ - (s - 1)

    c₀₀₀ = zero(S)
    c₀₀₁ = zero(S)
    c₀₁₀ = zero(S)
    c₀₁₁ = zero(S)
    c₁₀₀ = zero(S)
    c₁₀₁ = zero(S)
    c₁₁₀ = zero(S)
    c₁₁₁ = zero(S)

    @inbounds for k in 1:K
        ω = weights[k]
        c₀₀₀ = muladd(ω, policiesarray[i,     j,     s,     k], c₀₀₀)
        c₀₀₁ = muladd(ω, policiesarray[i,     j,     s + 1, k], c₀₀₁)
        c₀₁₀ = muladd(ω, policiesarray[i,     j + 1, s,     k], c₀₁₀)
        c₀₁₁ = muladd(ω, policiesarray[i,     j + 1, s + 1, k], c₀₁₁)
        c₁₀₀ = muladd(ω, policiesarray[i + 1, j,     s,     k], c₁₀₀)
        c₁₀₁ = muladd(ω, policiesarray[i + 1, j,     s + 1, k], c₁₀₁)
        c₁₁₀ = muladd(ω, policiesarray[i + 1, j + 1, s,     k], c₁₁₀)
        c₁₁₁ = muladd(ω, policiesarray[i + 1, j + 1, s + 1, k], c₁₁₁)
    end

    c₀₀ = muladd(ζ, c₀₀₁ - c₀₀₀, c₀₀₀)
    c₀₁ = muladd(ζ, c₀₁₁ - c₀₁₀, c₀₁₀)
    c₁₀ = muladd(ζ, c₁₀₁ - c₁₀₀, c₁₀₀)
    c₁₁ = muladd(ζ, c₁₁₁ - c₁₁₀, c₁₁₀)

    c₀ = muladd(η, c₀₁ - c₀₀, c₀₀)
    c₁ = muladd(η, c₁₁ - c₁₀, c₁₀)

    return muladd(ξ, c₁ - c₀, c₀)
end