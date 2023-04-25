# FIXME: How to deal with dimensions larger than 2
computestablemanifolds(F!, DF!, steadystates; kwargs...) = computestablemanifolds(F!, DF!, steadystates, Float64[]; kwargs...)
function computestablemanifolds(
	    F!::Function, DF!::Function,
	    steadystates::Vector{Vector{Float64}},
	    p::Vector{Any};
		alg = Tsit5(), abstol = 1.0e-10, reltol = 1.0e-10,
		h = 1e-3, tends = repeat([(10., 10.)], length(steadystates)), 
		T = 100,
		isoutofdomain = (u, p, t) -> false,
		verbose = false,
	    solverargs...)

	n = length(first(steadystates))

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
		
	for (j, x̄) ∈ enumerate(steadystates)
		verbose && println("Computing manifolds for steady state: ", x̄)

		J = zeros(n, n); DF!(J, x̄, p, 0.0)
		λ, V = eigen(J)

		stabledirs = findall(λᵢ -> real(λᵢ) < 0, λ)

		manifolds = Dict()

		for i ∈ stabledirs
			vᵢ = real.(V[:, i])

			negtend = tends[j][1]
			negprob = ODEProblem(odefn, x̄ - h * vᵢ, (0.0, negtend), p)
			negsol = solve(negprob, alg; reltol = reltol, abstol = abstol, isoutofdomain = isoutofdomain, solverargs...)

			if negsol.retcode == ReturnCode.Success
				timespan = range(0.0, negtend, length = T)
				manifolds[:n] = hcat((t -> negsol(negtend - t)).(timespan)...)'
			else
				manifolds[:n] = NaN * ones(T, n)
			end


			postend = tends[j][2]
			posprob = ODEProblem(odefn, x̄ + h * vᵢ, (0.0, postend), p)
			possol = solve(posprob, alg; reltol = reltol, abstol = abstol, isoutofdomain = isoutofdomain, solverargs...)

			if possol.retcode == ReturnCode.Success
				timespan = range(0.0, postend, length = T)
				manifolds[:p] = hcat((t -> possol(postend - t)).(timespan)...)'
			else
				manifolds[:p] = NaN * ones(T, n)
			end

		end		

		push!(equil, manifolds)
	end

	return equil
end