function computestablemanifolds(
	    F!::Function, DF!::Function,
	    steadystates::Vector{Vector{Float64}},
	    p::Vector{Any};
		alg = Tsit5(),
		h = 1e-3, T = 100,
		tends = repeat([(10., 10.)], length(steadystates)),
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

		manifolds = []

		for i ∈ stabledirs
			vᵢ = real.(V[:, i])

			# Negative direction
			negtend = tends[j][1]
			negprob = ODEProblem(odefn, x̄ - h * vᵢ, (0.0, negtend), p)
			negsol = solve(negprob, alg; solverargs...)

			if SciMLBase.successful_retcode(negsol.retcode)
				timespan = range(0.0, negtend, length = T)
				negmanifold = hcat((t -> negsol(negtend - t)).(timespan)...)'
			else
				negmanifold = NaN * ones(T, n)
			end

			# Positive direction
			postend = tends[j][2]
			posprob = ODEProblem(odefn, x̄ + h * vᵢ, (0.0, postend), p)
			possol = solve(posprob, alg; solverargs...)

			if SciMLBase.successful_retcode(possol.retcode)
				timespan = range(0.0, postend, length = T)
				posmanifold = hcat((t -> possol(postend - t)).(timespan)...)'
			else
				posmanifold = NaN * ones(T, n)
			end
			
			mᵢ = vcat(posmanifold, reverse(negmanifold, dims = 1))
			push!(manifolds, mᵢ)
		end		

		push!(equil, manifolds)
	end

	return equil
end