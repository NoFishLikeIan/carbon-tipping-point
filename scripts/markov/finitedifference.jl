"Takes a step backward in time `H(t) -> H(t - Δt)`."
function backwardstep!(problem, stencil, valuefunction::ValueFunction, Δt⁻¹, model::M, G::GR, calibration::Calibration; withnegative = false) where {N₁, N₂, S, M <: UnitIAM{S}, GR <: AbstractGrid{N₁,N₂,S}}
    # Consutruct the RHS
    
    # Construct the sparse LHS matrix
    problem.A .= constructA!(stencil, valuefunction, Δt⁻¹, model, G, calibration, withnegative)
    constructsource!(problem.b, valuefunction, Δt⁻¹, model, G, calibration)

    sol = solve!(problem)

    if !SciMLBase.successful_retcode(sol)
        throw("Time step solver failed at time $(valuefunction.t.t)!")
    end

    return sol
end

function updateovergrid!(H, u, θ)
    @inbounds for k in eachindex(u)
        H[k] = θ * u[k] + (1 - θ) * H[k]
    end
end

"Iterate linear solver until convergence"
function steadystate!(valuefunction::ValueFunction{S, N₁, N₂}, Δt::S, model::M, G::GR, calibration; timeiterations = 10_000, printstep = 100, tolerance::Error{S} = Error{S}(1e-3, 1e-3), verbose = 0, withnegative = false, alg = KLUFactorization(), θ = 1.) where {S, N₁, N₂, M <: UnitIAM{S}, GR <: AbstractGrid{N₁, N₂, S}}
    Δt⁻¹ = 1 / Δt
    
    # Initialise problem
    stencil = makestencil(G)
    A₀ = constructA!(stencil, valuefunction, Δt⁻¹, model, G, calibration, withnegative)
    b₀ = constructsource(valuefunction, Δt⁻¹, model, G, calibration)

    # Initialise the problem
    problem = LinearSolve.init(LinearProblem(A₀, b₀), alg)
    problem.u .= vec(valuefunction.H)
    sol = backwardstep!(problem, stencil, valuefunction, Δt⁻¹, model, G, calibration; withnegative)
    itererror = abserror(sol.u, valuefunction.H)
    
    for iter in 1:timeiterations  
        sol = backwardstep!(problem, stencil, valuefunction, Δt⁻¹, model, G, calibration; withnegative)
        itererror = abserror(sol.u, valuefunction.H)

        updateovergrid!(valuefunction.H, sol.u, θ)

        if itererror < tolerance
            return valuefunction, (iter, itererror)
        end

        if (verbose > 1) || (verbose > 0 && iter % printstep == 0)
            @printf "Iteration %d: absolute step = %.2e, relative step = %.2e\r" iter itererror.absolute itererror.relative
        end
    end

    @warn @sprintf "\nFailed convergence in %d iterations.\n" timeiterations

    return valuefunction, (timeiterations, itererror)
end

function backwardsimulation!(
    valuefunction::ValueFunction{S, N₁, N₂}, Δt::S, model::M, G::GR, calibration::Calibration; 
    t₀ = zero(S), withnegative = false,
    verbose = 0, printstep = S(10), 
    withsave = true, outdir = "data", overwrite = false,
    startcache = valuefunction.t.t, cachestep = S(1), alg = KLUFactorization()) where {S, N₁, N₂, M <: UnitIAM{S}, GR <: AbstractGrid{N₁, N₂, S}}
    
    if withsave
        cachepath, cachefile = initcachefile(model,G, outdir, withnegative; overwrite)
        tcache = copy(startcache)
        magnitude = -floor(Int, log10(abs(cachestep)))
        keyformat = Printf.Format("%.$(magnitude)f")
    end

    if verbose > 0
        tverbose = copy(valuefunction.t.t)
    end

    # Initialise problem
    # FIXME: Obsolete
    Δt⁻¹ = inv(Δt)
    source = constructsource(valuefunction, Δt⁻¹, model, G, calibration)
    adv = constructadv(valuefunction, model, G)
    stencil = makestencil(G)
    problemdata = (stencil, source, adv)

    A₀ = constructA!(stencil, valuefunction, Δt⁻¹, model, G, calibration, withnegative)
    b₀ = source - adv

    # Initialise the problem
    problem = LinearSolve.init(LinearProblem(A₀, b₀), alg)
    backwardstep!(problem, problemdata, valuefunction, Δt⁻¹, model, G, calibration; withnegative)
 
    while t₀ < valuefunction.t.t
        backwardstep!(problem, problemdata, valuefunction, Δt⁻¹, model, G, calibration; withnegative)
        updateovergrid!(valuefunction.H, problem.u, 1.)

        valuefunction.t.t -= Δt

        if (verbose > 1) || (verbose > 0 && valuefunction.t.t < tverbose)
            if verbose > 0 
                tverbose = tverbose - printstep 
            end
            @printf "Time %.2f\r" valuefunction.t.t
        end

        if withsave && valuefunction.t.t ≤ tcache
            cachekey = Printf.format(keyformat, tcache)
            if verbose > 1 @printf "\nSaving cache with key %s\n" cachekey end
            
            group = JLD2.Group(cachefile, cachekey)
            group["V"] = valuefunction

            tcache = tcache - cachestep
        end
    end

    if withsave
        close(cachefile)
        if verbose > 0 @printf "\nCached in %s\n" cachepath end
    end

    return valuefunction
end