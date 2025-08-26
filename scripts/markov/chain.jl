function εopt(t, Xᵢ::Point, ∂ₘH, model::M) where {S,D<:Damages{S},P<:LogSeparable{S},M<:AbstractModel{S,D,P}}
    @unpack preferences, economy, hogg = model

    dm = γ(t, calibration) + δₘ(exp(Xᵢ.m) * hogg.Mᵖ, hogg)
    ωₜ = ω(t, economy)

    return clamp(∂ₘH * dm / (ωₜ * (preferences.θ - 1)), 0, 5)
end


"Constructs upwind-downwind scheme A."
function constructA(V::ValueFunction, model::M, G::RegularGrid{N₁,N₂,S}, calibration::Calibration) where {N₁,N₂,S,D<:Damages{S},P<:LogSeparable{S},M<:AbstractModel{S,D,P}}
    n = length(G)
    ΔT⁻¹, Δm⁻¹ = inv.(step(G))
    t = V.t.t
    γₜ = γ(t, calibration)
    ωₜ = ω(t, economy)
    
    idx = Int[]; jdx = Int[]
    values = S[]

    @inbounds for j in axes(G, 2), i in axes(G, 1)
        k = LinearIndex((i, j), G)
        Xᵢ = G.X[k]
        δᵢ = δₘ(exp(Xᵢ.m) * hogg.Mᵖ, hogg)
 
        y = zero(S) # Diagonal values
        
        # Temperature, which is uncontrolled
        bᵀ = μ(Xᵢ.T, Xᵢ.m, model) / model.hogg.ϵ
        
        if bᵀ ≥ 0
            z = bᵀ * ΔT⁻¹
            push!(idx, k); push!(jdx, LinearIndex((i + 1, j), G))
            push!(values, z)
            y -= z
        else
            x = -bᵀ * ΔT⁻¹
            push!(idx, k); push!(jdx, LinearIndex((i - 1, j), G))
            push!(values, x)
            y -= x
        end

        # Carbon concentration is controlled
        if j < N₂
            ∂ᵐ₊H = (V.H[i, j + 1] - V.H[i, j]) * Δm⁻¹
            ε₊ = εopt(t, Xᵢ, ∂ᵐ₊H, model)
            bᵐ₊ = γₜ * (1 - ε₊) - δᵢ * ε₊
        else
            ε₊ = 1 / (1 + δᵢ / γₜ)
            ∂ᵐ₊H = ωₜ * (model.preferences.θ - 1) * ε₊ / (γₜ + δᵢ)
            bᵐ₊ = zero(S)
        end

        if j > 1
            ∂ᵐ₋H = (V.H[i, j] - V.H[i, j - 1]) * Δm⁻¹
            ε₋ = εopt(t, Xᵢ, ∂ᵐ₋H, model)
            bᵐ₋ = γₜ * (1 - ε₋) - δᵢ * ε₋
        else
            ε₋ = 1 / (1 + δᵢ / γₜ)
            ∂ᵐ₋H = ωₜ * (model.preferences.θ - 1) * ε₋ / (γₜ + δᵢ)
            bᵐ₋ = zero(S)
        end
        
        
        if bᵐ₊ > 0 && bᵐ₋ < 0
            error("Non-concave value function detected at grid point ($i, $j). " *
              "Forward drift bᵐ₊ = $bᵐ₊ > 0 and backward drift bᵐ₋ = $bᵐ₋ < 0, " *
              "which indicates the value function is not concave in the carbon concentration dimension.")
        elseif bᵐ₊ > 0 && bᵐ₋ ≥ 0
            V.ε[k] = ε₊

            z = bᵐ₊ * Δm⁻¹
            push!(idx, k); push!(jdx, LinearIndex((i, j + 1), G))
            push!(values, z)
            y -= z
        elseif bᵐ₊ ≤ 0 && bᵐ₋ < 0
            V.ε[k] = ε₋

            x = -bᵐ₋ * Δm⁻¹
            push!(idx, k); push!(jdx, LinearIndex((i, j - 1), G))
            push!(values, z)
            y -= z
        end

        push!(idx, k); push!(jdx, k)
        push!(values, y)
    end

    return sparse(idx, jdx, values, n, n)
end

"Constructs second order central difference operator."
function constructL(model::M, G::RegularGrid{N₁,N₂,S}) where {N₁,N₂,S,D<:Damages{S},P<:LogSeparable{S},M<:AbstractModel{S,D,P}}
    n = length(G)
    ΔT, Δm = step.(G.ranges)  # FIXED: Use step sizes, not extrema!
    νᵀ = (model.hogg.σₜ / (model.hogg.ϵ * ΔT))^2
    νᵐ = (model.hogg.σₘ / Δm)^2

    weights = S[]
    idx = Int[]
    jdx = []
    @inbounds for j in axes(G, 2), i in axes(G, 1)
        k = LinearIndex((i, j), G)

        # T direction second derivative
        if i > 1 && i < N₁
            push!(weights, νᵀ)
            push!(idx, k)
            push!(jdx, LinearIndex((i - 1, j), G))

            push!(weights, νᵀ)
            push!(idx, k)
            push!(jdx, LinearIndex((i + 1, j), G))

            push!(weights, -2νᵀ)
            push!(idx, k)
            push!(jdx, k)
        elseif i == 1
            # Left boundary: use forward difference
            push!(weights, νᵀ)
            push!(idx, k)
            push!(jdx, LinearIndex((i + 1, j), G))

            push!(weights, -νᵀ)
            push!(idx, k)
            push!(jdx, k)
        elseif i == N₁
            # Right boundary: use backward difference
            push!(weights, νᵀ)
            push!(idx, k)
            push!(jdx, LinearIndex((i - 1, j), G))

            push!(weights, -νᵀ)
            push!(idx, k)
            push!(jdx, k)
        end

        # m direction second derivative
        if j > 1 && j < N₂
            push!(weights, νᵐ)
            push!(idx, k)
            push!(jdx, LinearIndex((i, j - 1), G))

            push!(weights, νᵐ)
            push!(idx, k)
            push!(jdx, LinearIndex((i, j + 1), G))

            push!(weights, -2νᵐ)
            push!(idx, k)
            push!(jdx, k)
        elseif j == 1
            # Left boundary: use forward difference
            push!(weights, νᵐ)
            push!(idx, k)
            push!(jdx, LinearIndex((i, j + 1), G))

            push!(weights, -νᵐ)
            push!(idx, k)
            push!(jdx, k)
        elseif j == N₂
            # Right boundary: use backward difference
            push!(weights, νᵐ)
            push!(idx, k)
            push!(jdx, LinearIndex((i, j - 1), G))

            push!(weights, -νᵐ)
            push!(idx, k)
            push!(jdx, k)
        end
    end

    return sparse(idx, jdx, weights, n, n)
end