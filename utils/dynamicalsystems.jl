function isoutdomain(z::Vector{<:Real}, domain::Vector{Tuple{Float64, Float64}})
	for (i, zᵢ) ∈ enumerate(z)
		l, u = domain[i]
		if zᵢ < l || zᵢ > u
			return true
		end
	end

	return false
end

# FIXME: How to deal with dimensions larger than 2
computemanifolds(F!, DF!, steadystates; kwargs...) = computemanifolds(F!, DF!, steadystates, Float64[]; kwargs...)
	function computemanifolds(
	    F!::Function, DF!::Function,
	    steadystates::Vector{Vector{Float64}},
	    p::Vector{Any};
		alg = TanYam7(),
		abstol = 1.0e-8,
		reltol = 1.0e-8,
		h = 1e-3, dt = 1e-2,
		tend = 6.,
		domain = repeat([(-Inf, Inf)], length(first(steadystates))),
	    kwargs...)

		solver = (alg = alg, 
			abstol = abstol, reltol = reltol, 
			dt = dt, 
			isoutofdomain = (z, p, t) -> isoutdomain(z, domain)
		)
	
	    s = length(steadystates)
	    n = length(first(steadystates))

		odefn = ODEFunction(F!; jac = DF!)
	
		manifolds = []
	        
	    for (j, x̄) ∈ enumerate(steadystates)
	        J = zeros(n, n); DF!(J, x̄, p, 0.0)

	        λ, V = eigen(J)

			manifoldsofx̄ = []
	
	        for (i, vᵢ) ∈ enumerate(eachcol(V))
				isstable = real(λ[i]) < 0 # Stable if real part of eigenvalue is negative

				manifoldᵢ = NaN .* ones(2, T, n) # Stores left and right manifold
			
				if all(imag(vᵢ) .≈ 0) # TODO: How to handle imaginary eigenvalues?
					tspace = isstable ? (0., -tend) : (0., tend)
					timeframe = range(tspace...; length = T)

					u₀⁻ = x̄ - h * real.(vᵢ)
					u₀⁺ = x̄ + h * real.(vᵢ)
	
					prob⁻ = ODEProblem(odefn, u₀⁻, tspace, p)
					prob⁺ = ODEProblem(odefn, u₀⁺, tspace, p)
	
					sol⁻ = solve(prob⁻; solver...)
					sol⁺ = solve(prob⁺; solver...)
	
					for (k, t) ∈ enumerate(timeframe)
						manifoldᵢ[1, k, :] = sol⁻(t)
						manifoldᵢ[1, k, :] = sol⁺(t)
 					end
				end
				push!(manifoldsofx̄, (isstable, manifoldᵢ))
	        end

			push!(manifolds, manifoldsofx̄)
	    end
	
	    return manifolds
	end