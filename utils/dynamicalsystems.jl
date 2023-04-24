# FIXME: How to deal with dimensions larger than 2
computestablemanifolds(F!, DF!, steadystates; kwargs...) = computestablemanifolds(F!, DF!, steadystates, Float64[]; kwargs...)
	function computestablemanifolds(
	    F!::Function, DF!::Function,
	    steadystates::Vector{Vector{Float64}},
	    p::Vector{Any};
		alg = Tsit5(), abstol = 1.0e-8, reltol = 1.0e-8,
		h = 1e-3, tend = 6., T = 100,
		isoutofdomain = (u, p, t) -> false,
	    kwargs...)
	
	    n = length(first(steadystates))
		timespan = range(0.0, 2tend, length = T)

		function Finv!(dz, z, p, t)
			F!(dz, z, p, t)
			dz .= -dz
		end

		function DFinv!(J, z, p, t)
			DF!(J, z, p, t)
			J .= -J
		end

		odefn = ODEFunction(Finv!; jac = DFinv!)
		equil = []
	        
	    for x̄ ∈ steadystates
	        J = zeros(n, n); DF!(J, x̄, p, 0.0)
	        λ, V = eigen(J)

			stablemanifolds = []
	
	        for i ∈ 1:n
				v = V[:, i]
				isstable = real(λ[i]) < 0 # Stable if real part of eigenvalue is negative

				isvreal = all(imag(v) .≈ 0) # TODO: How to handle imaginary eigenvalues?

				if isstable && isvreal
					vᵣ = real.(v)
					manifold = NaN .* ones(2, T, n) # Stores left and right manifold
					
					negprob = ODEProblem(odefn, x̄ - h * vᵣ, (0.0, tend), p)
					negsol = solve(negprob, alg; reltol = reltol, abstol = abstol, isoutofdomain = isoutofdomain)
					negorbit = hcat((t -> negsol(tend - t)).(timespan)...)'
					
					manifold[1, :, :] = negorbit

					posprob = ODEProblem(odefn, x̄ + h * vᵣ, (0.0, tend), p)
					possol = solve(posprob, alg; reltol = reltol, abstol = abstol, isoutofdomain = isoutofdomain)
					posorbit = hcat((t -> possol(tend - t)).(timespan)...)'

					manifold[2, :, :] = posorbit

					push!(stablemanifolds, manifold)
				end
	        end

			push!(equil, x̄ => stablemanifolds)
	    end
	
	    return Dict(equil)
	end