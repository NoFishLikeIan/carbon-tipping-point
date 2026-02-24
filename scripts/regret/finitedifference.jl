"Updates stencils and source, assuming constant policy, and takes a step from `H(t)` to `H(t - Δt)`."
function backwardstep!(problem, weights, R, stencilm, regretfunction::ValueFunction, Δt⁻¹, model::M, G::GR, calibration::Calibration, policybasis::SimplexPolicies) where {N₁, N₂, S, M <: UnitIAM{S}, GR <: AbstractGrid{N₁,N₂,S}}
	# Construct the sparse LHS matrix
	n = length(G)
	constructDᵐ!(stencilm, weights, regretfunction, G, calibration, policybasis)
	problem.A = R - sparse(stencilm[1], stencilm[2], stencilm[3], n, n)
    
	# Consutruct the RHS
	constructsource!(problem.b, regretfunction, Δt⁻¹, model, G, calibration) # TODO: Modify this.
	sol = solve!(problem)

	if !SciMLBase.successful_retcode(sol)
		throw("Time step solver failed at time $(regretfunction.t.t)!")
	end

	return sol
end

"Iterate linear solver until convergence, assuming constant policies."
function steadystate!(weights, regretfunction::ValueFunction{S, N₁, N₂}, Δt::S, model::M, G::RegularGrid{N₁, N₂, S}, calibration, policybasis::SimplexPolicies; timeiterations = 10_000, printstep = 100, tolerance::Error{S} = Error{S}(1e-6, 1e-4), verbose = 0, alg = KLUFactorization()) where {S, N₁, N₂, M <: UnitIAM{S}}    
	# Initialise problem
	Δt⁻¹ = 1 / Δt
	n = length(G)
	stencilT, stencilm = makestencil(G)
	constructDᵀ!(stencilT, model, G)
	constructDᵐ!(stencilm, weights, regretfunction, G, calibration, policybasis)
	b₀ = constructsource(weights, regretfunction, Δt⁻¹, model, G, calibration, policybasis)
	Sᵨ = (preferences.ρ + Δt⁻¹) * I
	R = Sᵨ - sparse(stencilT[1], stencilT[2], stencilT[3], n, n)
	A₀ = R - sparse(stencilm[1], stencilm[2], stencilm[3], n, n)
	problem = LinearSolve.init(LinearProblem(A₀, b₀), alg)
    
	# First iteration
	backwardstep!(problem, R, stencilm, regretfunction, Δt⁻¹, model, G, calibration)
	itererror = abserror(problem.u, regretfunction.H)
	if itererror < tolerance return regretfunction, (1, itererror) end
    
	for iter in 2:timeiterations  
		backwardstep!(problem, R, stencilm, regretfunction, Δt⁻¹, model, G, calibration)
		itererror = abserror(problem.u, regretfunction.H)

		@inbounds for (k, uₖ) in enumerate(problem.u)
			regretfunction.H[k] = uₖ
		end

		if itererror < tolerance
			return regretfunction, (iter, itererror)
		end

		if (verbose > 1) || (verbose > 0 && iter % printstep == 0)
			@printf "Iteration %d: absolute step = %.2e, relative step = %.2e\r" iter itererror.absolute itererror.relative
		end
	end

	@warn @sprintf "\nFailed convergence in %d iterations.\n" timeiterations

	return regretfunction, (timeiterations, itererror)
end

function setpolicy!(valuefunction::V, abatement::P, G::GR) where {V <: ValueFunction, P <: Interpolations.AbstractInterpolation, GR <: RegularGrid}
	Tspace, mspace = G.ranges
	@inbounds for j in axes(G, 2), i in axes(G, 1)
		valuefunction.α[i, j] = abatement(Tspace[i], mspace[j], valuefunction.t.t)
	end
end
"Backward simulation of `valuefunction` from `valuefunction.t` to `t₀`, assuming an exogenous `abatement` policy stored as a matrix with `(T, m, t)`. Returns a `OrderedDict` with either starting and terminal `valuefunction`, if `storetrajectory` is `false`, or the whole trajectory, otherwise."
function exogenousbackwardsimulation!(valuefunction::ValueFunction{S, N₁, N₂}, abatement::P, Δt::S, model::M, G::GR, calibration::Calibration; t₀ = zero(S), verbose = 0, printstep = 10, withsave = true, outdir = "data", overwrite = false, startcache = valuefunction.t.t, cachestep = one(S), alg = KLUFactorization(), storetrajectory = false) where {S, N₁, N₂, M <: UnitIAM{S}, GR <: AbstractGrid{N₁, N₂, S}, P <: Interpolations.AbstractInterpolation}
	tcache = copy(startcache)
	valuefunctiontraj = OrderedDict(valuefunction.t.t => copy(valuefunction))

	if withsave
		cachepath, cachefile = initcachefile(model,G, outdir, withnegative; overwrite)
		magnitude = -floor(Int, log10(abs(cachestep)))
		keyformat = Printf.Format("%.$(magnitude)f")
	end

	if verbose > 0
		tverbose = copy(valuefunction.t.t)
	end

	# Initialise problem
	Δt⁻¹ = 1 / Δt
	n = length(G)
	stencilT, stencilm = makestencil(G)
	constructDᵀ!(stencilT, model, G)
	constructexogenousDᵐ!(stencilm, valuefunction, G, calibration)
	b₀ = constructsource(valuefunction, Δt⁻¹, model, G, calibration)
	Sᵨ = (preferences.ρ + Δt⁻¹) * I
	R = Sᵨ - sparse(stencilT[1], stencilT[2], stencilT[3], n, n)
	A₀ = R - sparse(stencilm[1], stencilm[2], stencilm[3], n, n)
	problem = LinearSolve.init(LinearProblem(A₀, b₀), alg)
    
	# First iteration
	exogenousbackwardstep!(problem, R, stencilm, valuefunction, Δt⁻¹, model, G, calibration)
 
	while t₀ < valuefunction.t.t
		valuefunction.t.t -= Δt
		setpolicy!(valuefunction, abatement, G)
		exogenousbackwardstep!(problem, R, stencilm, valuefunction, Δt⁻¹, model, G, calibration)

		@inbounds for (k, uₖ) in enumerate(problem.u)
			valuefunction.H[k] = uₖ
		end

		if (verbose > 1) || (verbose > 0 && valuefunction.t.t < tverbose)
			if verbose > 0 
				tverbose = tverbose - printstep 
			end
			@printf "Time %.2f\r" valuefunction.t.t
		end

		if valuefunction.t.t ≤ tcache
			if withsave
				cachekey = Printf.format(keyformat, tcache)
				if verbose > 1
					@printf "\nSaving cache with key %s\n" cachekey
				end
                
				group = JLD2.Group(cachefile, cachekey)
				group["V"] = valuefunction
			end

			if storetrajectory
				valuefunctiontraj[valuefunction.t.t] = copy(valuefunction)
			end

			tcache = tcache - cachestep
		end
	end

	if withsave
		close(cachefile)
		if verbose > 0 @printf "\nCached in %s\n" cachepath end
	end

	valuefunctiontraj[valuefunction.t.t] = copy(valuefunction)
	sort!(valuefunctiontraj)
    
	return valuefunctiontraj
end
