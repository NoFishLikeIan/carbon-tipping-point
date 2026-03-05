function constructDᵐ!(stencil::StencilData, weights, t::Time, G::RegularGrid{N₁, N₂, S}, calibration::Calibration, policybasis::SimplexPolicies) where {N₁, N₂, S}
	γₜ = γ(t, calibration)
    
	Δm = step(G, 2)
	Tspace, mspace = G.ranges
	rows, columns, data = stencil
	counter = 1
	@inbounds for i in axes(G, 1), j in axes(G, 2)
		k = LinearIndex((i, j), G)
		x = Point(Tspace[i], mspace[j])

		y = zero(S) # Diagonal values

		αₖ = weightedpolicy(x, t.t, weights, policybasis)
		bᵐ = (γₜ - αₖ) / Δm

		if 1 < j < N₂
			z = max(bᵐ, 0)
			rows[counter] = k; columns[counter] = LinearIndex((i, j + 1), G)
			data[counter] = z; counter += 1

			x = max(-bᵐ, 0)
			rows[counter] = k; columns[counter] = LinearIndex((i, j - 1), G)
			data[counter] = x; counter += 1

			y -= (x + z)
		elseif j == 1 # Lower boundary
			z = max(bᵐ, 0)
			rows[counter] = k; columns[counter] = LinearIndex((i, 2), G)
			data[counter] = z; counter += 1

			y -= z
		else # Upper boundary
			x = max(-bᵐ, 0)
			rows[counter] = k; columns[counter] = LinearIndex((i, N₂ - 1), G)
			data[counter] = x; counter += 1

			y -= x
		end

		rows[counter] = k; columns[counter] = k;
		data[counter] = y;

		counter += 1
	end
end

"Constructs source vector `Δt⁻¹ Hⁿ + b`."
function constructsource(weights::W, valuefunction::ValueFunction, Δt⁻¹, model::M, G::GR, calibration, policybasis::SimplexPolicies) where {N₁, N₂, S, M <: UnitIAM, GR <: AbstractGrid{N₁, N₂, S}, TW, W <: AbstractVector{TW}}
    constructsource!(Vector{TW}(undef, N₁ * N₂), weights, valuefunction, Δt⁻¹, model, G, calibration, policybasis)
end
"Updates source vector `Δt⁻¹ Hⁿ + b`."
function constructsource!(source, weights, valuefunction::ValueFunction, Δt⁻¹, model::M, G::GR, calibration, policybasis::SimplexPolicies) where {N₁, N₂, S, M <: UnitIAM, GR <: AbstractGrid{N₁, N₂, S}}
    @unpack t, H, α = valuefunction
    Tspace, mspace = G.ranges
    @inbounds for i in axes(G, 1), j in axes(G, 2)
        x = Point(Tspace[i], mspace[j])
        αₖ = weightedpolicy(x, t.t, weights, policybasis)
        Hₖ = H[i, j]

        ΔT = step(G, 1)
        bᵀ = μ(x.T, x.m, model.climate)
        useforward = (bᵀ ≥ 0 && i < N₁) || (bᵀ < 0 && i == 1) 

        ∂ᵀH = ((useforward ? H[i + 1, j] : H[i - 1, j]) - Hₖ) / ΔT
        advection = ∂ᵀH^2 * variance(x.T, model.climate.hogg) / 2

        k = LinearIndex((i, j), G)
        source[k] = advection + l(t.t, x, αₖ, model, calibration) + Δt⁻¹ * Hₖ
    end

    return source
end