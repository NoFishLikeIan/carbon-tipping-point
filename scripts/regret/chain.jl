"Constructs upwind-downwind scheme discretiser `Dᵐ` for CO₂e log-concentration `m`, conditional a policy function `α`, which is not updated."
function constructexogenousDᵐ!(stencil::StencilData{S}, valuefunction::ValueFunction{S, N₁, N₂}, G::RegularGrid{N₁, N₂, S}, calibration::Calibration) where {N₁, N₂, S}
	@unpack t, H, α = valuefunction
	γₜ = γ(t, calibration)
    
	Δm = step(G, 2)
	Tspace, mspace = G.ranges
	rows, columns, data = stencil
	counter = 1
	@inbounds for j in axes(G, 2), i in axes(G, 1)
		k = LinearIndex((i, j), G)
		x = Point(Tspace[i], mspace[j])

		y = zero(S) # Diagonal values

		αₖ = α[k]
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
