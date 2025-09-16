function αopt(t, Xᵢ::Point, ∂ₘH, model::M, calibration::Calibration) where {S, M <: UnitIAM{S}}
    @unpack economy, preferences = model

    if ∂ₘH ≤ 0 return zero(S) end

    @unpack abatement, investments = economy
    b = abatement.b

    num = ∂ₘH * ᾱ(t, Xᵢ, model, calibration)^b
    den = b * A(t, investments) * ω(t, abatement) * (preferences.θ - 1)

    return (num / den)^inv(b - 1)
end
function implicit∂ₘH(t, Xᵢ::Point, αᵢ, model::M, calibration::Calibration) where {S, M <: UnitIAM{S}}
    @unpack economy, preferences = model
    
    if αᵢ ≤ 0 return zero(αᵢ) end
    
    b = economy.abatement.b
    
    return αᵢ^(b - 1) * b * A(t, economy.investments) * ω(t, economy.abatement) * (preferences.θ - 1) / ᾱ(t, Xᵢ, model, calibration)^b
end

function upperbound(t, Xᵢ, model::M, calibration::Calibration, withnegative) where {S, M <: UnitIAM{S}}
    ifelse(withnegative, 2, 1) * ᾱ(t, Xᵢ, model, calibration)
end
function stencilsize(::GR) where {N₁, N₂, GR <: AbstractGrid{N₁, N₂}}
    7 * N₁ * N₂ - 2N₂
end
function makestencil(G::GR) where {N₁, N₂, S, GR <: AbstractGrid{N₁, N₂, S}}
    n = stencilsize(G)
    idx = Vector{Int}(undef, n)
    jdx = Vector{Int}(undef, n)
    data = Vector{S}(undef, n)

    return (idx, jdx, data)
end

StencilData{S} = Tuple{Vector{Int}, Vector{Int}, Vector{S}}
"Constructs upwind-downwind scheme discretiser `(ρ + Δt⁻¹)I - L - B(Hⁿ)` and updates policy accordingly."
function constructA!(V, Δt⁻¹, model, G::GR, calibration, withnegative) where {N₁, N₂, S, GR <: AbstractGrid{N₁, N₂, S}}
    constructA!(makestencil(G), V, Δt⁻¹, model, G, calibration, withnegative)
end
function constructA!(stencil::StencilData{S}, V::ValueFunction{S, N₁, N₂}, Δt⁻¹, model::M, G::GR, calibration::Calibration, withnegative) where {N₁, N₂, S, M <: UnitIAM{S}, GR <: AbstractGrid{N₁, N₂, S}}
    (idx, jdx, data) = stencil

    t = V.t.t
    γₜ = γ(t, calibration)
    r = model.preferences.ρ + Δt⁻¹

    n = length(G)
    
    entrydx = 1
    @inbounds for j in axes(G, 2), i in axes(G, 1)
        k = LinearIndex((i, j), G)
        Xᵢ = G[k]
        σₜ² = variance(Xᵢ.T, model.climate.hogg)

        (ΔT₋, ΔT₊), (Δm₋, Δm₊) = steps(G, i, j)
 
        y = zero(S) # Diagonal values
        
        # Temperature, which is uncontrolled
        bᵀ = μ(Xᵢ.T, Xᵢ.m, model.climate) / model.climate.hogg.ϵ
        if (bᵀ ≥ 0 && i < N₁) || (bᵀ < 0 && i == 1)
            ∂ᵀH = (V.H[i + 1, j] - V.H[i, j]) / ΔT₊

            z = bᵀ / ΔT₊
            x = zero(S)
        else
            ∂ᵀH = (V.H[i, j] - V.H[i - 1, j]) / ΔT₋
            
            z = zero(S)
            x = -bᵀ / ΔT₋
        end

        zdx = min(i + 1, N₁)
        idx[entrydx] = k; jdx[entrydx] = LinearIndex((zdx, j), G); data[entrydx] = -z
        entrydx += 1
        xdx = max(i - 1, 1)
        idx[entrydx] = k; jdx[entrydx] = LinearIndex((xdx, j), G); data[entrydx] = -x
        entrydx += 1

        y -= (z + x)

        # Temperature noise terms (second derivative)
        if i > 1 && i < N₁
            # Central difference for second derivative with variable spacing
            νT₋ = σₜ² / (ΔT₋ * (ΔT₋ + ΔT₊))
            νT₊ = σₜ² / (ΔT₊ * (ΔT₋ + ΔT₊))
            
            idx[entrydx] = k; jdx[entrydx] = LinearIndex((i - 1, j), G); data[entrydx] = -νT₋
            entrydx += 1
            idx[entrydx] = k; jdx[entrydx] = LinearIndex((i + 1, j), G); data[entrydx] = -νT₊
            entrydx += 1
            y -= (νT₋ + νT₊)
        elseif i == 1
            νT = (σₜ² / 2) / ΔT₊^2
            idx[entrydx] = k; jdx[entrydx] = LinearIndex((i + 1, j), G); data[entrydx] = -νT
            entrydx += 1
            y -= νT
        elseif i == N₁
            νT = (σₜ² / 2) / ΔT₋^2
            idx[entrydx] = k; jdx[entrydx] = LinearIndex((i - 1, j), G); data[entrydx] = -νT
            entrydx += 1
            y -= νT
        end

        # Carbon concentration is controlled
        αmax = upperbound(t, Xᵢ, model, calibration, withnegative)
        
        # Forward direction (positive drift)
        if j < N₂
            ∂ᵐ₊H = (V.H[i, j + 1] - V.H[i, j]) / Δm₊
            α₊ = clamp(αopt(t, Xᵢ, ∂ᵐ₊H, model, calibration), 0, αmax)
        else
            α₊ = γₜ
            ∂ᵐ₊H = implicit∂ₘH(t, Xᵢ, α₊, model, calibration)
        end

        bᵐ₊ = γₜ - α₊

        if j > 1
            ∂ᵐ₋H = (V.H[i, j] - V.H[i, j - 1]) / Δm₋
            α₋ = clamp(αopt(t, Xᵢ, ∂ᵐ₋H, model, calibration), 0, αmax)
        else
            α₋ = γₜ
            ∂ᵐ₋H = implicit∂ₘH(t, Xᵢ, α₋, model, calibration)
        end
        
        bᵐ₋ = γₜ - α₋

        if bᵐ₊ > 0 && bᵐ₋ < 0 # Drifts are discordant outwards, use Hamiltonian
            H₊ = l(t, Xᵢ, α₊, model, calibration) + ∂ᵐ₊H * bᵐ₊
            H₋ = l(t, Xᵢ, α₋, model, calibration) + ∂ᵐ₋H * bᵐ₋

            if H₊ < H₋ # Use upward derivative
                V.α[k] = α₊
                z = bᵐ₊ / Δm₊
                x = zero(S)
            else # Use downard derivative
                V.α[k] = α₋
                z = zero(S)
                x = -bᵐ₋ / Δm₋
            end
        elseif bᵐ₊ > 0 && bᵐ₋ ≥ 0 # Drifts agree upward
            V.α[k] = α₊
            z = bᵐ₊ / Δm₊
            x = zero(S)
        elseif bᵐ₊ ≤ 0 && bᵐ₋ < 0 # Drifts agree downard
            V.α[k] = α₋
            z = zero(S)
            x = -bᵐ₋ / Δm₋
        elseif bᵐ₊ ≤ 0 && bᵐ₋ ≥ 0 # Drifts disagree inwards, use steady state
            V.α[k] = γₜ
            z = zero(S)
            x = zero(S)
        else
            throw("Drift combination not implemented: sign(bᵐ₊)=$(sign(bᵐ₊)), sign(bᵐ₋)=$(sign(bᵐ₋))")
        end

        zdx = min(j + 1, N₂)
        idx[entrydx] = k; jdx[entrydx] = LinearIndex((i, zdx), G); data[entrydx] = -z
        entrydx += 1
        
        xdx = max(j - 1, 1)
        idx[entrydx] = k; jdx[entrydx] = LinearIndex((i, xdx), G); data[entrydx] = -x
        entrydx += 1
        y -= (z + x)


        idx[entrydx] = k; jdx[entrydx] = k; data[entrydx] = r - y
        entrydx += 1
    end

    return SparseArrays.sparse(idx, jdx, data, n, n)
end

"Constructs source vector `Δt⁻¹ Hⁿ + b`."
function constructsource(valuefunction::ValueFunction, Δt⁻¹, model::M, G::GR, calibration) where {N₁, N₂, S, M <: UnitIAM, GR <: AbstractGrid{N₁, N₂, S}}
    constructsource!(Vector{S}(undef, length(G)), valuefunction, Δt⁻¹, model, G, calibration)
end
"Updates source vector `Δt⁻¹ Hⁿ + b`."
function constructsource!(source, valuefunction::ValueFunction, Δt⁻¹, model::M, G::GR, calibration) where {N₁, N₂, S, M <: UnitIAM, GR <: AbstractGrid{N₁, N₂, S}}
    @unpack H, α = valuefunction
    @inbounds for j in axes(G, 2), i in axes(G, 1)
        Xᵢ = G[i, j]
        αᵢ = valuefunction.α[i, j]
        Hᵢ = valuefunction.H[i, j]

        (ΔT₋, ΔT₊), _ = steps(G, i, j)
        bᵀ = μ(Xᵢ.T, Xᵢ.m, model.climate) / model.climate.hogg.ϵ
        ∂ᵀH = (bᵀ ≥ 0 && i < N₁) || (bᵀ < 0 && i == 1) ? (H[i + 1, j] - H[i, j]) / ΔT₊ : (H[i - 1, j] - H[i, j]) / ΔT₋

        k = LinearIndex((i, j), G)
        source[k] = l(valuefunction.t.t, Xᵢ, αᵢ, model, calibration) + Δt⁻¹ * Hᵢ + ∂ᵀH^2 * variance(Xᵢ.T, model.climate.hogg) / 2
    end

    return source
end

function centralderivative(valuefunction::ValueFunction{S, N₁, N₂}, G::GR) where {N₁, N₂, S, GR <: AbstractGrid{N₁, N₂, S}}
    @unpack H = valuefunction

    ∂Hₘ = similar(H)

    @inbounds for j in axes(H, 2), i in axes(H, 1)
        _, (Δm₋, Δm₊) = steps(G, i, j)
        
        ∂Hₘ[i, j] = if j == 1
            (H[i, j + 1] - H[i, j]) / Δm₊
        elseif j == size(H, 2)
            (H[i, j] - H[i, j - 1]) / Δm₋
        else
            (H[i, j + 1] - H[i, j - 1]) / (Δm₋ + Δm₊)
        end
    end

    return ∂Hₘ

end

function centralpolicy(valuefunction::ValueFunction{S, N₁, N₂}, model::M, G::GR, calibration; withnegative = false) where {N₁, N₂, S, M <: UnitIAM{S}, GR <: AbstractGrid{N₁, N₂, S}}
    α = similar(valuefunction.H)
    centralpolicy!(α, valuefunction, model, G, calibration; withnegative)
end
function centralpolicy!(valuefunction::ValueFunction{S, N₁, N₂}, model::M, G::GR, calibration; withnegative = false) where {N₁, N₂, S, M <: UnitIAM{S}, GR <: AbstractGrid{N₁, N₂, S}}
    centralpolicy!(valuefunction.α, valuefunction, model, G, calibration; withnegative)
end
function centralpolicy!(α, valuefunction::ValueFunction{S, N₁, N₂}, model::M, G::GR, calibration; withnegative = false) where {N₁, N₂, S, M <: UnitIAM{S}, GR <: AbstractGrid{N₁, N₂, S}}
    @unpack H = valuefunction

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

    return α
end