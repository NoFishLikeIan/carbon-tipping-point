function αopt(t, Xᵢ::Point, ∂ₘH, model::M, calibration::Calibration) where {S, M <: UnitElasticityModel{S}}
    -∂ₘH * ᾱ(t, Xᵢ, model, calibration)^2 / (A(t, economy) * ω(t, model.economy) * (1 - model.preferences.θ))
end

"Constructs upwind-downwind scheme A."
function constructA(V::ValueFunction, Δt⁻¹, model::M, G::RegularGrid{N₁,N₂,S}, calibration::Calibration; withnegative = false) where {N₁, N₂, S, M <: UnitElasticityModel{S}}
    ΔT⁻¹, Δm⁻¹ = inversestep(G)
    ΔT⁻² = ΔT⁻¹^2
    Δm⁻² = Δm⁻¹^2

    σₜ² = (model.hogg.σₜ / model.hogg.ϵ)^2 / 2
    σₘ² = model.hogg.σₘ^2 / 2

    t = V.t.t
    γₜ = γ(t, calibration)
    ωₜ = ω(t, economy)
    r = model.preferences.ρ + Δt⁻¹

    idx = Int[]; jdx = Int[]; values = S[]
    @inbounds for j in axes(G, 2), i in axes(G, 1)
        k = LinearIndex((i, j), G)
        Xᵢ = G[k]
 
        y = zero(S) # Diagonal values
        
        # Temperature, which is uncontrolled
        bᵀ = μ(Xᵢ.T, Xᵢ.m, model) / model.hogg.ϵ     
        if bᵀ ≥ 0
            ∂ᵀH = (i < N₁ ? V.H[i + 1, j] - V.H[i, j] : V.H[i, j] - V.H[i - 1, j]) * ΔT⁻¹

            z = bᵀ * ΔT⁻¹ + 2σₜ² * ∂ᵀH
            
            push!(idx, k); push!(jdx, LinearIndex((i + 1, j), G))
            push!(values, -z)
            y -= z
        else
            ∂ᵀH = (i > 1 ? V.H[i, j] - V.H[i - 1, j] : V.H[i + 1, j] - V.H[i, j]) * ΔT⁻¹

            x = -bᵀ * ΔT⁻¹ + 2σₜ² * ∂ᵀH
            
            push!(idx, k); push!(jdx, LinearIndex((i - 1, j), G))
            push!(values, -x)
            y -= x
        end

        # Temperature noise terms (second derivative)
        νT = σₜ² * ΔT⁻²
        if i > 1 && i < N₁
            push!(idx, k); push!(jdx, LinearIndex((i - 1, j), G))
            push!(values, -νT)
            push!(idx, k); push!(jdx, LinearIndex((i + 1, j), G))
            push!(values, -νT)
            y -= 2νT
        elseif i == 1
            push!(idx, k); push!(jdx, LinearIndex((i + 1, j), G))
            push!(values, -νT)
            y -= νT
        elseif i == N₁
            push!(idx, k); push!(jdx, LinearIndex((i - 1, j), G))
            push!(values, -νT)
            y -= νT
        end

        # Carbon concentration is controlled
        αmax = ifelse(withnegative, one(S), ᾱ(t, Xᵢ, model, calibration))
        if j < N₂
            ∂ᵐ₊H = (V.H[i, j + 1] - V.H[i, j]) * Δm⁻¹
            α₊ = clamp(αopt(t, Xᵢ, ∂ᵐ₊H, model, calibration), 0, αmax)
            bᵐ₊ = γₜ - α₊
        else
            α₊ = ᾱ(t, Xᵢ, model, calibration)
            ∂ᵐ₊H = γₜ * ωₜ * (model.preferences.θ - 1) / α₊^2
            bᵐ₊ = zero(S)
        end

        if j > 1
            ∂ᵐ₋H = (V.H[i, j] - V.H[i, j - 1]) * Δm⁻¹
            α₋ = clamp(αopt(t, Xᵢ, ∂ᵐ₋H, model, calibration), 0, αmax)
            bᵐ₋ = γₜ - α₋
        else
            α₋ = ᾱ(t, Xᵢ, model, calibration)
            ∂ᵐ₋H = γₜ * ωₜ * (model.preferences.θ - 1) / α₋^2
            bᵐ₋ = zero(S)
        end
        
        if bᵐ₊ ≥ 0 && bᵐ₋ ≤ 0
            H₊ = l(t, Xᵢ, α₊, model, calibration) + ∂ᵐ₊H * bᵐ₊
            H₋ = l(t, Xᵢ, α₋, model, calibration) + ∂ᵐ₋H * bᵐ₋

            if H₊ < H₋ # Minimisation problem
                V.α[k] = α₊
                
                z = bᵐ₊ * Δm⁻¹ + 2σₘ² * ∂ᵐ₊H
                push!(idx, k); push!(jdx, LinearIndex((i, j + 1), G))
                push!(values, -z)
                y -= z
            else
                V.α[k] = α₋

                x = -bᵐ₋ * Δm⁻¹ + 2σₘ² * ∂ᵐ₋H
                push!(idx, k); push!(jdx, LinearIndex((i, j - 1), G))
                push!(values, -x)
                y -= x
            end
            
        elseif bᵐ₊ > 0 && bᵐ₋ ≥ 0
            V.α[k] = α₊

            z = bᵐ₊ * Δm⁻¹ + 2σₘ² * ∂ᵐ₊H
            push!(idx, k); push!(jdx, LinearIndex((i, j + 1), G))
            push!(values, -z)
            y -= z
        elseif bᵐ₊ ≤ 0 && bᵐ₋ < 0
            V.α[k] = α₋

            x = -bᵐ₋ * Δm⁻¹ + 2σₘ² * ∂ᵐ₋H
            push!(idx, k); push!(jdx, LinearIndex((i, j - 1), G))
            push!(values, -x)
            y -= x
        end

        # Carbon concentration noise terms (second derivative)
        νm = σₘ² * Δm⁻²
        if j > 1 && j < N₂
            push!(idx, k); push!(jdx, LinearIndex((i, j - 1), G))
            push!(values, -νm)
            push!(idx, k); push!(jdx, LinearIndex((i, j + 1), G))
            push!(values, -νm)
            y -= 2νm
        elseif j == 1
            push!(idx, k); push!(jdx, LinearIndex((i, j + 1), G))
            push!(values, -νm)
            y -= νm
        elseif j == N₂
            push!(idx, k); push!(jdx, LinearIndex((i, j - 1), G))
            push!(values, -νm)
            y -= νm
        end

        push!(idx, k); push!(jdx, k)
        push!(values, r - y)
    end

    n = length(G)
    return sparse(idx, jdx, values, n, n)
end

function constructb(valuefunction::ValueFunction, Δt⁻¹, model::M, G::RegularGrid{N₁,N₂,S}, calibration) where {N₁, N₂, S, M <: UnitElasticityModel{S}}
    constructb!(Vector{S}(undef, length(G)), valuefunction, Δt⁻¹, model, G, calibration)
end
function constructb!(b, valuefunction::ValueFunction, Δt⁻¹, model::M, G::RegularGrid{N₁,N₂,S}, calibration) where {N₁, N₂, S, M <: UnitElasticityModel{S}}
    ΔT⁻¹, Δm⁻¹ = inv.(step(G))
    γₜ = γ(valuefunction.t.t, calibration)

    @inbounds for j in axes(G, 2), i in axes(G, 1)
        k = LinearIndex((i, j), G)

        Xᵢ = G[k]
        αᵢ = valuefunction.α[k]

        bᵀ = μ(Xᵢ.T, Xᵢ.m, model) / model.hogg.ϵ
        ∂ᵀH = if (bᵀ ≥ 0 && i < N₁) || (bᵀ < 0 && i == 1)
            (valuefunction.H[i + 1, j] - valuefunction.H[i, j]) * ΔT⁻¹
        else
           (valuefunction.H[i, j] - valuefunction.H[i - 1, j]) * ΔT⁻¹
        end

        bᵐ = γₜ - valuefunction.α[k]
        ∂ᵐH = if (bᵐ ≥ 0 && j < N₂) || (bᵐ < 0 && j == 1)
            (valuefunction.H[i, j + 1] - valuefunction.H[i, j]) * Δm⁻¹
        else
           (valuefunction.H[i, j] - valuefunction.H[i, j - 1]) * Δm⁻¹
        end

        adv = ∂ᵀH * (model.hogg.σₜ / model.hogg.ϵ)^2 + ∂ᵐH * model.hogg.σₘ^2

        b[k] = l(valuefunction.t.t, Xᵢ, αᵢ, model, calibration) + Δt⁻¹ * valuefunction.H[k] + adv
    end

    return b
end

function centralpolicy!(valuefunction::ValueFunction{S, N₁, N₂}, model::M, G::RegularGrid{N₁, N₂, S}, calibration) where {N₁, N₂, S, M <: UnitElasticityModel{S}}
    @unpack H, α = valuefunction
    Δm⁻¹ = inversestep(G)[2]

    @inbounds for j in axes(H, 2), i in axes(H, 1)
        ∂ₘH = (
            if j == 1
                H[i, j + 1] - H[i, j]
            elseif j == size(H, 2)
                H[i, j] - H[i, j - 1]
            else
                (H[i, j + 1] - H[i, j - 1]) / 2
            end
        ) * Δm⁻¹
        
        α[i, j] = αopt(valuefunction.t.t, G[i, j], ∂ₘH, model, calibration)
    end

    return valuefunction
end