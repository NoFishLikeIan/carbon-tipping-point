computemanifolds(F!, DF!, steadystates; kwargs...) = computemanifolds(F!, DF!, steadystates, Float64[]; kwargs...)
	function computemanifolds(
	    F!::Function, DF!::Function,
	    steadystates::Vector{Vector{Float64}},
	    p::Vector{Any};
	    T = 100, limit = 1000, dt = 0.01, h = 1e-3,
	    kwargs...)
	
	    m = length(steadystates)
	    n = length(first(steadystates))
	
	    function Finv!(dx, x, p, t)
	        F!(dx, x, p, t)
	        dx .= -dx
	    end
	
	    fwds = CoupledODEs(F!, zeros(n), p)
	    bckds = CoupledODEs(Finv!, zeros(n), p)
	
	    manifolds = NaN * ones(m, 2, 2, T, n)
	        
	    for (j, x̄) ∈ enumerate(steadystates)
	        J = zeros(n, n); DF!(J, x̄, p, 0.0)
	        λ, V = eigen(J)
	
	        for (i, vᵢ) ∈ enumerate(eachcol(V))
	            isstable = real(λ[i]) < 0 # Stable if real part of eigenvalue is negative
	
	            ds = isstable ? bckds : fwds
	            
	            for (k, op) ∈ enumerate([-, +])
	                reinit!(ds, op(x̄, vᵢ * h))
	
	                for t ∈ 1:T
	                    step!(ds, dt)
	                    xₙ = ds.integ.u
	
	                    manifolds[j, i, k, t, :] = xₙ
	
	                    if norm(xₙ) > limit break end
	                end         
	            end
	        end
	    end
	
	    return manifolds
	end