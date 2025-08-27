function αopt(t, Xᵢ::Point, ∂ₘH, model::M, calibration::Calibration) where {S,D<:Damages{S},P<:LogSeparable{S},M<:AbstractModel{S,D,P}}
    αnet = ᾱ(t, Xᵢ, model, calibration)

    return clamp(-∂ₘH * αnet^2 / (ω(t, model.economy) * (1 - model.preferences.θ)), 0, αnet)
end

"Constructs upwind-downwind scheme A."
function constructD(V::ValueFunction, model::M, G::RegularGrid{N₁,N₂,S}, calibration::Calibration) where {N₁,N₂,S,D<:Damages{S},P<:LogSeparable{S},M<:AbstractModel{S,D,P}}
    n = length(G)
    ΔT⁻¹, Δm⁻¹ = inv.(step(G))
    ΔT⁻² = ΔT⁻¹^2
    Δm⁻² = Δm⁻¹^2

    σₜ² = (model.hogg.σₜ / model.hogg.ϵ)^2 / 2
    σₘ² = model.hogg.σₘ^2 / 2

    t = V.t.t
    γₜ = γ(t, calibration)
    ωₜ = ω(t, economy)
    
    idx = Int[]; jdx = Int[]
    values = S[]

    for j in axes(G, 2), i in axes(G, 1)
        k = LinearIndex((i, j), G)
        Xᵢ = G.X[k]
 
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

        # Temperature noise terms (second derivative)
        νT = σₜ² * ΔT⁻²
        if i > 1 && i < N₁
            push!(idx, k); push!(jdx, LinearIndex((i - 1, j), G))
            push!(values, νT)
            push!(idx, k); push!(jdx, LinearIndex((i + 1, j), G))
            push!(values, νT)
            y -= 2νT
        elseif i == 1
            push!(idx, k); push!(jdx, LinearIndex((i + 1, j), G))
            push!(values, νT)
            y -= νT
        elseif i == N₁
            push!(idx, k); push!(jdx, LinearIndex((i - 1, j), G))
            push!(values, νT)
            y -= νT
        end

        # Carbon concentration is controlled
        if j < N₂
            ∂ᵐ₊H = (V.H[i, j + 1] - V.H[i, j]) * Δm⁻¹
            α₊ = αopt(t, Xᵢ, ∂ᵐ₊H, model, calibration)
            bᵐ₊ = γₜ - α₊
        else
            α₊ = ᾱ(t, Xᵢ, model, calibration)
            ∂ᵐ₊H = ωₜ * (model.preferences.θ - 1) / α₊
            bᵐ₊ = zero(S)
        end

        if j > 1
            ∂ᵐ₋H = (V.H[i, j] - V.H[i, j - 1]) * Δm⁻¹
            α₋ = αopt(t, Xᵢ, ∂ᵐ₋H, model, calibration)
            bᵐ₋ = γₜ - α₋
        else
            α₋ = ᾱ(t, Xᵢ, model, calibration)
            ∂ᵐ₋H = ωₜ * (model.preferences.θ - 1) / α₋
            bᵐ₋ = zero(S)
        end
        
        if bᵐ₊ ≥ 0 && bᵐ₋ ≤ 0

            H₊ = l(t, Xᵢ, α₊, model, calibration) + ∂ᵐ₊H * bᵐ₊
            H₋ = l(t, Xᵢ, α₋, model, calibration) + ∂ᵐ₋H * bᵐ₋

            if H₊ < H₋ # Minimisation problem
                z = bᵐ₊ * Δm⁻¹
                push!(idx, k); push!(jdx, LinearIndex((i, j + 1), G))
                push!(values, z)
                y -= z
            else
                V.α[k] = α₋

                x = -bᵐ₋ * Δm⁻¹
                push!(idx, k); push!(jdx, LinearIndex((i, j - 1), G))
                push!(values, x)
                y -= x
            end
            
        elseif bᵐ₊ > 0 && bᵐ₋ ≥ 0
            V.α[k] = α₊

            z = bᵐ₊ * Δm⁻¹
            push!(idx, k); push!(jdx, LinearIndex((i, j + 1), G))
            push!(values, z)
            y -= z
        elseif bᵐ₊ ≤ 0 && bᵐ₋ < 0
            V.α[k] = α₋

            x = -bᵐ₋ * Δm⁻¹
            push!(idx, k); push!(jdx, LinearIndex((i, j - 1), G))
            push!(values, x)
            y -= x
        end


        # Carbon concentration noise terms (second derivative)
        νm = σₘ² * Δm⁻²
        if j > 1 && j < N₂
            push!(idx, k); push!(jdx, LinearIndex((i, j - 1), G))
            push!(values, νm)
            push!(idx, k); push!(jdx, LinearIndex((i, j + 1), G))
            push!(values, νm)
            y -= 2νm
        elseif j == 1
            push!(idx, k); push!(jdx, LinearIndex((i, j + 1), G))
            push!(values, νm)
            y -= νm
        elseif j == N₂
            push!(idx, k); push!(jdx, LinearIndex((i, j - 1), G))
            push!(values, νm)
            y -= νm
        end

        push!(idx, k); push!(jdx, k)
        push!(values, y)
    end

    return sparse(idx, jdx, values, n, n)
end

"Constructs second order central difference operator."
function constructL(model::M, G::RegularGrid{N₁,N₂,S}) where {N₁,N₂,S,D<:Damages{S},P<:LogSeparable{S},M<:AbstractModel{S,D,P}}
    n = length(G)
    ΔT, Δm = step.(G.ranges)
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

function constructA(valuefunction::ValueFunction, Δt⁻¹, model::M, G::RegularGrid{N₁,N₂,S}, calibration::Calibration) where {N₁,N₂,S,D<:Damages{S},P<:LogSeparable{S},M<:AbstractModel{S,D,P}}
    (model.preferences.ρ + Δt⁻¹) * I - constructD(valuefunction, model, G, calibration)
end