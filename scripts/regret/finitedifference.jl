function backwardstep!(problem, weights, R, stencilm, valuefunction::ValueFunction, Δt⁻¹, model::M, G::GR, calibration::Calibration, policybasis::SimplexPolicies) where {N₁, N₂, S, M <: UnitIAM{S}, GR <: AbstractGrid{N₁,N₂,S}}
	# Construct the sparse LHS matrix
	n = length(G)
	constructDᵐ!(stencilm, weights, valuefunction, G, calibration, policybasis)
	problem.A = R - sparse(stencilm[1], stencilm[2], stencilm[3], n, n)
    
	# Consutruct the RHS
	constructsource!(problem.b, weights, valuefunction, Δt⁻¹, model, G, calibration, policybasis)
	sol = solve!(problem)

	if !SciMLBase.successful_retcode(sol)
		throw("Time step solver failed at time $(valuefunction.t.t)!")
	end

	return sol
end

function steadystate!(valuefunction::ValueFunction{S, N₁, N₂}, weights::W, Δt::S, model::M, G::RegularGrid{N₁, N₂, S}, calibration, policybasis::SimplexPolicies; timeiterations = 10_000, printstep = 100, tolerance::Error{S} = Error{S}(1e-6, 1e-4), verbose = 0, alg = KLUFactorization()) where {S, K, TW, N₁, N₂, M <: UnitIAM{S}, W <: StaticArray{K, TW}}
	
	Δt⁻¹ = 1 / Δt
	n = length(G)

	stencilT, stencilm = makestencil(TW, G)
	
	constructDᵀ!(stencilT, model, G)
	constructDᵐ!(stencilm, weights, valuefunction.t, G, calibration, policybasis)
	
	b₀ = constructsource(weights, valuefunction, Δt⁻¹, model, G, calibration, policybasis)
	Sᵨ = (model.preferences.ρ + Δt⁻¹) * I
	R = Sᵨ - sparse(stencilT[1], stencilT[2], stencilT[3], n, n)
	A₀ = R - sparse(stencilm[1], stencilm[2], stencilm[3], n, n)
	
	problem = LinearSolve.init(LinearProblem(A₀, b₀), alg)
    
	# First iteration
	sol = solve!(problem)
	
	if !SciMLBase.successful_retcode(sol)
		throw("Steady state solver failed at first iteration!")
	end
	
	itererror = abserror(problem.u, valuefunction.H)
	if itererror < tolerance
		copyto!(valuefunction.H, problem.u)
		return valuefunction, (1, itererror), problem
	end
    
	for iter in 2:timeiterations  
		constructsource!(problem.b, weights, valuefunction, Δt⁻¹, model, G, calibration, policybasis)
		sol = solve!(problem)
		
		if !SciMLBase.successful_retcode(sol)
			throw("Steady state solver failed at iteration $iter!")
		end
		
		itererror = abserror(problem.u, valuefunction.H)
		copyto!(valuefunction.H, problem.u)

		if itererror < tolerance
			return valuefunction, (iter, itererror), problem
		end

		if (verbose > 1) || (verbose > 0 && iter % printstep == 0)
			@printf "Iteration %d: absolute step = %.2e, relative step = %.2e\r" iter itererror.absolute itererror.relative
		end
	end

	@warn @sprintf "\nFailed convergence in %d iterations.\n" timeiterations

	return valuefunction, (timeiterations, itererror)
end

function backwardsimulation!(valuefunction::ValueFunction{S, N₁, N₂}, weights::W, Δt::S, model::M, G::GR, calibration::Calibration, policybasis::SimplexPolicies; t₀ = zero(S), verbose = 0, printstep = 10, alg = KLUFactorization(), storetrajectory = false, startcache = valuefunction.t.t, cachestep = one(S)) where {S, K, TW, N₁, N₂, M <: UnitIAM{S}, GR <: AbstractGrid{N₁, N₂, S}, W <: StaticArray{K, TW}}
	tcache = copy(startcache)
	valuefunctiontraj = OrderedDict(valuefunction.t.t => copy(valuefunction))

	if verbose > 0
		tverbose = copy(valuefunction.t.t)
	end

	# Initialise problem
	Δt⁻¹ = 1 / Δt
	n = length(G)
	stencilT, stencilm = makestencil(TW, G)
	constructDᵀ!(stencilT, model, G)
	constructDᵐ!(stencilm, weights, valuefunction, G, calibration, policybasis)
	b₀ = constructsource(weights, valuefunction, Δt⁻¹, model, G, calibration, policybasis)
	Sᵨ = (preferences.ρ + Δt⁻¹) * I
	R = Sᵨ - sparse(stencilT[1], stencilT[2], stencilT[3], n, n)
	A₀ = R - sparse(stencilm[1], stencilm[2], stencilm[3], n, n)
	problem = LinearSolve.init(LinearProblem(A₀, b₀), alg)
    
	# First iteration
	backwardstep!(problem, weights, R, stencilm, valuefunction, Δt⁻¹, model, G, calibration, policybasis)
 
	while t₀ < valuefunction.t.t
		valuefunction.t.t -= Δt
		backwardstep!(problem, weights, R, stencilm, valuefunction, Δt⁻¹, model, G, calibration, policybasis)

		copyto!(valuefunction.H, problem.u)

		if (verbose > 1) || (verbose > 0 && valuefunction.t.t < tverbose)
			if verbose > 0 
				tverbose = tverbose - printstep 
			end
			@printf "Time %.2f\r" valuefunction.t.t
		end

		if valuefunction.t.t ≤ tcache

			if storetrajectory
				valuefunctiontraj[valuefunction.t.t] = copy(valuefunction)
			end

			tcache = tcache - cachestep
		end
	end

	valuefunctiontraj[valuefunction.t.t] = copy(valuefunction)
	sort!(valuefunctiontraj)
    
	return valuefunctiontraj
end