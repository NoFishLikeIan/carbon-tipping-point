"Optimal abatement policy"
function φ(t, x::Point, ∂ₘH, model::M, calibration::Calibration) where {S, M <: UnitIAM{S}}
    @unpack economy, preferences = model

    if ∂ₘH ≤ 0 return zero(S) end

    @unpack abatement, investments = economy
    @unpack b = abatement

    num = ∂ₘH * ᾱ(t, x, model, calibration)^b
    den = b * A(t, investments) * ω(t, abatement) * (preferences.θ - 1)

    return (num / den)^inv(b - 1)
end
function φ⁻¹(t, x::Point, αᵢ, model::M, calibration::Calibration) where {S, M <: UnitIAM{S}}
    @unpack economy, preferences = model

    if αᵢ ≤ 0 return zero(αᵢ) end
    
    @unpack abatement, investments = economy
    @unpack b = abatement
    
    return αᵢ^(b - 1) * b * A(t, investments) * ω(t, abatement) * (preferences.θ - 1) / ᾱ(t, x, model, calibration)^b
end

function upperbound(t, x, model::M, calibration::Calibration, withnegative) where {S, M <: UnitIAM{S}}
    ifelse(withnegative, Inf, 1) * ᾱ(t, x, model, calibration)
end
function stencilsizes(::GR) where {N₁, N₂, GR <: AbstractGrid{N₁, N₂}}
    nᵀ = N₂ * (3N₁ - 2)
    nᵐ = N₁ * (3N₂ - 2)

    return (nᵀ, nᵐ)
end
function makestencil(G::GR) where {N₁, N₂, S, GR <: AbstractGrid{N₁, N₂, S}}
    n = stencilsizes(G)
    stencils = ntuple(j -> begin
            rows = Vector{Int}(undef, n[j])
            columns = Vector{Int}(undef, n[j])
            data = Vector{S}(undef, n[j])

            (rows, columns, data)
        end, 2)

    return stencils
end
function makeequilibriumstencil(::GR) where {N₁, N₂, S, GR <: AbstractGrid{N₁, N₂, S}}
    n = 3N₂ - 2 
    rows = Vector{Int}(undef, n)
    columns = Vector{Int}(undef, n)
    data = Vector{S}(undef, n)

    return (rows, columns, data)
end

StencilData{S} = Tuple{Vector{Int}, Vector{Int}, Vector{S}}
"Constructs upwind-downwind scheme discretiser `Dᵀ` for temperature `T`."
function constructDᵀ!(stencil::StencilData{S}, model::M, G::RegularGrid{N₁, N₂, S}) where {N₁, N₂, S, M <: UnitIAM{S}}
    ΔT = step(G, 1)
    Tspace, mspace = G.ranges

    rows, columns, data = stencil
    counter = 1
    @inbounds for j in axes(G, 2), i in axes(G, 1)
        k = LinearIndex((i, j), G)
        x = Point(Tspace[i], mspace[j])
        
        y = zero(S) # Diagonal values
        
        σₜ² = variance(x.T, model.climate.hogg) / ΔT^2
        bᵀ = μ(x.T, x.m, model.climate) / (model.climate.hogg.ϵ * ΔT)

        if 1 < i < N₁
            z = σₜ² / 2 + max(bᵀ, 0)
            rows[counter] = k; columns[counter] = LinearIndex((i + 1, j), G)
            data[counter] = z; counter += 1
            
            x = σₜ² / 2 + max(-bᵀ, 0)
            rows[counter] = k; columns[counter] = LinearIndex((i - 1, j), G)
            data[counter] = x; counter += 1

            y -= (x + z)
        elseif i == 1 # Lower boundary
            z = max(bᵀ, 0) + σₜ²
            rows[counter] = k; columns[counter] = LinearIndex((2, j), G)
            data[counter] = z; counter += 1

            y -= z
        else # Upper boundary
            x = max(-bᵀ, 0) + σₜ²
            rows[counter] = k; columns[counter] = LinearIndex((N₁ - 1, j), G)
            data[counter] = x; counter += 1

            y -= x
        end

        rows[counter] = k; columns[counter] = k;
        data[counter] = y;

        counter += 1
    end
end
"Constructs upwind-downwind scheme discretiser `Dᵐ` for CO2e log-concentration `m` and updates policy `α`."
function constructDᵐ!(stencil::StencilData{S}, valuefunction::ValueFunction{S, N₁, N₂}, model::M, G::RegularGrid{N₁, N₂, S}, calibration::Calibration, withnegative::Bool) where {N₁, N₂, S, M <: UnitIAM{S}}
    @unpack t, H, α = valuefunction
    γₜ = γ(t.t, calibration)
    
    Δm = step(G, 2)
    Tspace, mspace = G.ranges
    rows, columns, data = stencil
    counter = 1
    @inbounds for j in axes(G, 2), i in axes(G, 1)
        k = LinearIndex((i, j), G)
        x = Point(Tspace[i], mspace[j])
        αmax = upperbound(t.t, x, model, calibration, withnegative)

        # Choose consistent drift / derivative pair
        if j < N₂
            ∂ᵐ₊H = (H[i, j + 1] - H[i, j]) / Δm
            α₊ = min(φ(t.t, x, ∂ᵐ₊H, model, calibration), αmax)
        else
            α₊ = γₜ
            ∂ᵐ₊H = φ⁻¹(t.t, x, α₊, model, calibration)
        end

        if j > 1
            ∂ᵐ₋H = (H[i, j] - H[i, j - 1]) / Δm
            α₋ = min(φ(t.t, x, ∂ᵐ₋H, model, calibration), αmax)
        else
            α₋ = γₜ
            ∂ᵐ₋H = φ⁻¹(t.t, x, α₋, model, calibration)
        end

        bᵐ₊ = (γₜ - α₊) / Δm
        bᵐ₋ = (γₜ - α₋) / Δm

        if bᵐ₊ > 0 && bᵐ₋ ≥ 0 # Drifts agree forward
            α[i, j] = α₊
            z = bᵐ₊
            x = zero(S)
        elseif bᵐ₋ < 0 && bᵐ₊ ≤ 0 # Drifts agree backward
            α[i, j] = α₋
            z = zero(S)
            x = -bᵐ₋
        elseif bᵐ₊ ≤ 0 && bᵐ₋ ≥ 0 # Drifts point inwards, steady-state
            α[i, j] = γₜ
            z = zero(S)
            x = zero(S)
        elseif bᵐ₊ > 0 && bᵐ₋ < 0 # Drifts point outwards, non-concavity, follow Hamiltonian
            H₊ = l(t.t, x, α₊, model, calibration) + ∂ᵐ₊H * bᵐ₊ * Δm
            H₋ = l(t.t, x, α₋, model, calibration) + ∂ᵐ₋H * bᵐ₋ * Δm

            if H₊ < H₋
                α[i, j] = α₊
                z = bᵐ₊
                x = zero(S)
            else
                α[i, j] = α₋
                z = zero(S)
                x = -bᵐ₋
            end
        else throw("Unhandled pair of drifts: bᵐ₊ = $(bᵐ₊), bᵐ₋ = $(bᵐ₋)") end

        if j < N₂
            rows[counter] = k; columns[counter] = LinearIndex((i, j + 1), G)
            data[counter] = z; counter += 1
        end

        if j > 1
            rows[counter] = k; columns[counter] = LinearIndex((i, j - 1), G)
            data[counter] = x; counter += 1
        end

        rows[counter] = k; columns[counter] = k
        data[counter] = -(x + z); counter += 1
    end
end
"Constructs the one-dimensional discretiser `Dᵐ` for CO2e log-concentration `m` and updates policy `α`, under `ϵ → 0`. Requires climate model to be linear."
function constructequilibriumDᵐ!(equilibriumstencil::StencilData{S}, (t, H, α), model::M, G::RegularGrid{N₁, N₂, S}, calibration::Calibration, withnegative::Bool) where {N₁, N₂, S, D, P, C <: LinearClimate, M <: UnitIAM{S, D, P, C}}
    _, mspace = G.ranges
	Δm = step(mspace)
    γₜ = γ(t, calibration)

    rows, columns, data = equilibriumstencil
    counter = 1
	@inbounds for j in axes(G, 2)
		m = mspace[j]
        T = Tstable(m, model.climate) |> only
        x = Point(T, m)

        if j < N₂
            ∂ᵐ₊H = (H[j + 1] - H[j]) / Δm
            α₊ = φ(t, x, ∂ᵐ₊H, model, calibration)
        else
            α₊ = γₜ
            ∂ᵐ₊H = φ⁻¹(t, x, α₊, model, calibration)
        end
        
		if j > 1
            ∂ᵐ₋H = (H[j] - H[j - 1]) / Δm
            α₋ = φ(t, x, ∂ᵐ₋H, model, calibration)
        else
            α₋ = γₜ
            ∂ᵐ₋H = φ⁻¹(t, x, α₋, model, calibration)
        end

        bᵐ₊ = (γₜ - α₊) / Δm
        bᵐ₋ = (γₜ - α₋) / Δm
		
        if bᵐ₊ > 0 && bᵐ₋ ≥ 0 # Drifts agree forward
            α[j] = α₊
            z = bᵐ₊
            x = zero(S)
        elseif bᵐ₋ < 0 && bᵐ₊ ≤ 0 # Drifts agree backward
            α[j] = α₋
            z = zero(S)
            x = -bᵐ₋
        elseif bᵐ₊ ≤ 0 && bᵐ₋ ≥ 0 # Drifts point inwards, steady-state
            α[j] = γₜ
            z = zero(S)
            x = zero(S)
        elseif bᵐ₊ > 0 && bᵐ₋ < 0 # Drifts point outwards, non-concavity, follow Hamiltonian
            H₊ = l(t, x, α₊, model, calibration) + ∂ᵐ₊H * bᵐ₊ * Δm
            H₋ = l(t, x, α₋, model, calibration) + ∂ᵐ₋H * bᵐ₋ * Δm

            if H₊ < H₋
                α[j] = α₊
                z = bᵐ₊
                x = zero(S)
            else
                α[j] = α₋
                z = zero(S)
                x = -bᵐ₋
            end
        else throw("Unhandled pair of drifts: bᵐ₊ = $(bᵐ₊), bᵐ₋ = $(bᵐ₋)") end

        if j < N₂
            rows[counter] = j; columns[counter] = j + 1
            data[counter] = z; counter += 1
        end

        if j > 1
            rows[counter] = j; columns[counter] = j - 1
            data[counter] = x; counter += 1
        end

        rows[counter] = j; columns[counter] = j
        data[counter] = -(x + z); counter += 1
	end
end

"Constructs source vector `Δt⁻¹ Hⁿ + b`."
function constructsource(valuefunction::ValueFunction, Δt⁻¹, model::M, G::GR, calibration) where {N₁, N₂, S, M <: UnitIAM, GR <: AbstractGrid{N₁, N₂, S}}
    constructsource!(Vector{S}(undef, N₁ * N₂), valuefunction, Δt⁻¹, model, G, calibration)
end
"Updates source vector `Δt⁻¹ Hⁿ + b`."
function constructsource!(source, valuefunction::ValueFunction, Δt⁻¹, model::M, G::GR, calibration) where {N₁, N₂, S, M <: UnitIAM, GR <: AbstractGrid{N₁, N₂, S}}
    @unpack t, H, α = valuefunction
    Tspace, mspace = G.ranges
    @inbounds for j in axes(G, 2), i in axes(G, 1)
        x = Point(Tspace[i], mspace[j])
        αᵢ = α[i, j]
        Hᵢ = H[i, j]

        ΔT = step(G, 1)
        bᵀ = μ(x.T, x.m, model.climate)
        useforward = (bᵀ ≥ 0 && i < N₁) || (bᵀ < 0 && i == 1) 

        ∂ᵀH = ((useforward ? H[i + 1, j] : H[i - 1, j]) - Hᵢ) / ΔT
        advection = ∂ᵀH^2 * variance(x.T, model.climate.hogg) / 2

        k = LinearIndex((i, j), G)
        source[k] = advection + l(t.t, x, αᵢ, model, calibration) + Δt⁻¹ * Hᵢ
    end

    return source
end

function constructequilibriumsource((t, H, α), Δt⁻¹, model::M, G::GR, calibration) where {N₁, N₂, S, D, P, C <: LinearClimate, M <: UnitIAM{S, D, P, C}, GR <: AbstractGrid{N₁, N₂, S}}
    constructequilibriumsource!(Vector{S}(undef, N₂), (t, H, α), Δt⁻¹, model, G, calibration)
end
"Updates source vector `Δt⁻¹ Hⁿ + b`."
function constructequilibriumsource!(equilibriumsource, (t, H, α), Δt⁻¹, model::M, G::GR, calibration) where {N₁, N₂, S, D, P, C <: LinearClimate, M <: UnitIAM{S, D, P, C}, GR <: AbstractGrid{N₁, N₂, S}}
    mspace = G.ranges[2]

    @inbounds for j in axes(G, 2)
        m = mspace[j]
        T = Tstable(m, model.climate) |> only
        x = Point(T, m)

        αᵢ = α[j]
        Hᵢ = H[j]

        equilibriumsource[j] = l(t, x, αᵢ, model, calibration) + Δt⁻¹ * Hᵢ
    end

    return equilibriumsource
end