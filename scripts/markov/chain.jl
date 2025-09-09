function αopt(t, Xᵢ::Point, ∂ₘH, model::M, calibration::Calibration) where {S, M <: UnitElasticityModel{S}}
    -∂ₘH * ᾱ(t, Xᵢ, model, c
alibration)^2 / (A(t, economy) * ω(t, model.economy) * (1 - model.preferences.θ))
end

function upperbound(t, Xᵢ, model, ca economy.abatement.b *libration, withnegative)
    ifelse(withnegative, 1.5, 1) * ᾱ(t, Xᵢ, model, calibration)
end

"Constructs upwind-downwind scheme discretiser `(ρ + Δt⁻¹)I - L - B(Hⁿ)` and updates policy accordingly."
function constructA!(V::ValueFunction{S, N₁, N₂}, Δt⁻¹, model::M, G::GR, calibration::Calibration, withnegative) where {N₁, N₂, S, M <: UnitElasticityModel{S}, GR <: AbstractGrid{N₁, N₂, S}}
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

        (ΔT₋, ΔT₊), (Δm₋, Δm₊) = steps(G, i, j)
 
        y = zero(S) # Diagonal values
        
        # Temperature, which is uncontrolled
        bᵀ = μ(Xᵢ.T, Xᵢ.m, model) / model.hogg.ϵ

        if (bᵀ ≥ 0 && i < N₁) || (bᵀ < 0 && i == 1)
            ∂ᵀH = (V.H[i + 1, j] - V.H[i, j]) / ΔT₊

            z = bᵀ / ΔT₊ + 2σₜ² * ∂ᵀH
            x = zero(S)
        else
            ∂ᵀH = (V.H[i, j] - V.H[i - 1, j]) / ΔT₋
            
            z = zero(S)
            x = -bᵀ / ΔT₋ + 2σₜ² * ∂ᵀH
        end

        zdx = min(i + 1, N₁)
        push!(idx, k); push!(jdx, LinearIndex((zdx, j), G))
        push!(values, -z)
        
        xdx = max(i - 1, 1)
        push!(idx, k); push!(jdx, LinearIndex((xdx, j), G))
        push!(values, -x)
        y -= (z + x)

        # Temperature noise terms (second derivative)
        if i > 1 && i < N₁
            # Central difference for second derivative with variable spacing
            νT₋ = 2σₜ² / (ΔT₋ * (ΔT₋ + ΔT₊))
            νT₊ = 2σₜ² / (ΔT₊ * (ΔT₋ + ΔT₊))
            
            push!(idx, k); push!(jdx, LinearIndex((i - 1, j), G))
            push!(values, -νT₋)
            push!(idx, k); push!(jdx, LinearIndex((i + 1, j), G))
            push!(values, -νT₊)
            y -= (νT₋ + νT₊)
        elseif i == 1
            νT = σₜ² / ΔT₊^2
            push!(idx, k); push!(jdx, LinearIndex((i + 1, j), G))
            push!(values, -νT)
            y -= νT
        elseif i == N₁
            νT = σₜ² / ΔT₋^2
            push!(idx, k); push!(jdx, LinearIndex((i - 1, j), G))
            push!(values, -νT)
            y -= νT
        end

        # Carbon concentration is controlled
        αmax = upperbound(t, Xᵢ, model, calibration, withnegative)
        if j < N₂
            ∂ᵐ₊H = (V.H[i, j + 1] - V.H[i, j]) / Δm₊
            α₊ = clamp(αopt(t, Xᵢ, ∂ᵐ₊H, model, calibration), 0, αmax)
            bᵐ₊ = γₜ - α₊
        else
            α₊ = ᾱ(t, Xᵢ, model, calibration)
            ∂ᵐ₊H = γₜ * ωₜ * (model.preferences.θ - 1) / α₊^2
            bᵐ₊ = zero(S)
        end

        if j > 1
            ∂ᵐ₋H = (V.H[i, j] - V.H[i, j - 1]) / Δm₋
            α₋ = clamp(αopt(t, Xᵢ, ∂ᵐ₋H, model, calibration), 0, αmax)
            bᵐ₋ = γₜ - α₋
        else
            α₋ = ᾱ(t, Xᵢ, model, calibration)
            ∂ᵐ₋H = γₜ * ωₜ * (model.preferences.θ - 1) / α₋^2
            bᵐ₋ = zero(S)
        end
        
        if bᵐ₊ ≥ 0 && bᵐ₋ ≤ 0
            H₊ = l(t, Xᵢ, α₊, model, calibration) + ∂ᵐ₊H * bᵐ₊
            H₋ = l(t, Xᵢ, α₋, model, calibration) + ∂ᵐ₋H * bᵐ₋

            if H₊ < H₋ # Minimisation problem
                V.α[k] = α₊         
                z = bᵐ₊ / Δm₊ + 2σₘ² * ∂ᵐ₊H
                x = zero(S)
            else
                V.α[k] = α₋
                z = zero(S)
                x = -bᵐ₋ / Δm₋ + 2σₘ² * ∂ᵐ₋H
            end        
        elseif bᵐ₊ > 0 && bᵐ₋ ≥ 0
            V.α[k] = α₊

            z = bᵐ₊ / Δm₊ + 2σₘ² * ∂ᵐ₊H
            x = zero(S)
        elseif bᵐ₊ ≤ 0 && bᵐ₋ < 0
            V.α[k] = α₋

            x = -bᵐ₋ / Δm₋ + 2σₘ² * ∂ᵐ₋H
            z = zero(S)
        end

        zdx = min(j + 1, N₂)
        push!(idx, k); push!(jdx, LinearIndex((i, zdx), G))
        push!(values, -z)
        
        xdx = max(j - 1, 1)
        push!(idx, k); push!(jdx, LinearIndex((i, xdx), G))
        push!(values, -x)
        y -= (z + x)

        # Carbon concentration noise terms (second derivative)
        if j > 1 && j < N₂
            # Central difference for second derivative with variable spacing
            νm₋ = 2σₘ² / (Δm₋ * (Δm₋ + Δm₊))
            νm₊ = 2σₘ² / (Δm₊ * (Δm₋ + Δm₊))
            
            push!(idx, k); push!(jdx, LinearIndex((i, j - 1), G))
            push!(values, -νm₋)
            push!(idx, k); push!(jdx, LinearIndex((i, j + 1), G))
            push!(values, -νm₊)
            y -= (νm₋ + νm₊)
        elseif j == 1
            νm = σₘ² / Δm₊^2
            push!(idx, k); push!(jdx, LinearIndex((i, j + 1), G))
            push!(values, -νm)
            y -= νm
        elseif j == N₂
            νm = σₘ² / Δm₋^2
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

"Constructs source vector `Δt⁻¹ Hⁿ + b`."
function constructsource(valuefunction::ValueFunction, Δt⁻¹, model::M, G::GR, calibration) where {N₁, N₂, S, M <: UnitElasticityModel, GR <: AbstractGrid{N₁, N₂, S}}
    constructsource!(Vector{S}(undef, length(G)), valuefunction, Δt⁻¹, model, G, calibration)
end
"Replaces source vector `Δt⁻¹ Hⁿ + b`."
function constructsource!(source, valuefunction::ValueFunction, Δt⁻¹, model::M, G::GR, calibration) where {N₁, N₂, S, M <: UnitElasticityModel, GR <: AbstractGrid{N₁, N₂, S}}
    @inbounds for j in axes(G, 2), i in axes(G, 1)
        k = LinearIndex((i, j), G)

        Xᵢ = G[k]
        αᵢ = valuefunction.α[k]

        source[k] = l(valuefunction.t.t, Xᵢ, αᵢ, model, calibration) + Δt⁻¹ * valuefunction.H[k]
    end

    return source
end

"Constructs advection coefficient `adv(Hⁿ)`."
function constructadv(valuefunction::ValueFunction, model::M, G::GR, calibration) where {N₁, N₂, S, M <: UnitElasticityModel, GR <: AbstractGrid{N₁, N₂, S}}
    constructadv!(Vector{S}(undef, length(G)), valuefunction, model, G, calibration)
end
"Constructs advection coefficient `adv(Hⁿ)`."
function constructadv!(adv, valuefunction::ValueFunction, model::M, G::GR, calibration) where {N₁, N₂, S, M <: UnitElasticityModel, GR <: AbstractGrid{N₁, N₂, S}}
    γₜ = γ(valuefunction.t.t, calibration)

    @inbounds for j in axes(G, 2), i in axes(G, 1)
        k = LinearIndex((i, j), G)

        Xᵢ = G[k]
        αᵢ = valuefunction.α[k]

        (ΔT₋, ΔT₊), (Δm₋, Δm₊) = steps(G, i, j)

        bᵀ = μ(Xᵢ.T, Xᵢ.m, model) / model.hogg.ϵ
        ∂ᵀH = if (bᵀ ≥ 0 && i < size(G, 1)) || (bᵀ < 0 && i == 1)
            (valuefunction.H[i + 1, j] - valuefunction.H[i, j]) / ΔT₊
        else
           (valuefunction.H[i, j] - valuefunction.H[i - 1, j]) / ΔT₋
        end

        bᵐ = γₜ - αᵢ
        ∂ᵐH = if (bᵐ ≥ 0 && j < N₂) || (bᵐ < 0 && j == 1)
            (valuefunction.H[i, j + 1] - valuefunction.H[i, j]) / Δm₊
        else
           (valuefunction.H[i, j] - valuefunction.H[i, j - 1]) / Δm₋
        end

        adv[k] = (∂ᵀH * model.hogg.σₜ / model.hogg.ϵ)^2 + (∂ᵐH * model.hogg.σₘ)^2
    end

    return adv
end

function centralpolicy!(valuefunction::ValueFunction{S, N₁, N₂}, model::M, G::GR, calibration; withnegative = false) where {N₁, N₂, S, M <: UnitElasticityModel{S}, GR <: AbstractGrid{N₁, N₂, S}}
    @unpack H, α = valuefunction

    @inbounds for j in axes(H, 2), i in axes(H, 1)
        (ΔT₋, ΔT₊), (Δm₋, Δm₊) = steps(G, i, j)
        
        ∂ₘH = (
            if j == 1
                (H[i, j + 1] - H[i, j]) / Δm₊
            elseif j == size(H, 2)
                (H[i, j] - H[i, j - 1]) / Δm₋
            else
                (H[i, j + 1] - H[i, j - 1]) / (Δm₋ + Δm₊)
            end
        )
        
        αmax = upperbound(valuefunction.t.t, G[i, j], model, calibration, withnegative)

        α[i, j] = clamp(αopt(valuefunction.t.t, G[i, j], ∂ₘH, model, calibration), 0, αmax)
    end

    return valuefunction
end