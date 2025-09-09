function picardstep!(problem, source, valuefunction::ValueFunction, Δt⁻¹, model::M, G::GR, calibration::Calibration; withnegative = false) where {N₁, N₂, S, M <: UnitIAM{S}, GR <: AbstractGrid{N₁,N₂,S}}
    problem.b .= source - constructadv(valuefunction, model, G,  calibration)
    problem.A = constructA!(valuefunction, Δt⁻¹, model, G, calibration, withnegative)

    sol = solve!(problem)
    
    if !SciMLBase.successful_retcode(sol)
        throw("Picard step solver failed at time $(valuefunction.t.t)!")
    end

    return problem
end

"Takes a step backward in time `H(t) -> H(t - Δt)`."
function backwardstep!(problem, valuefunction::ValueFunction, Δt⁻¹, model::M, G::GR, calibration::Calibration; withnegative = false) where {N₁, N₂, S, M <: UnitIAM{S}, GR <: AbstractGrid{N₁,N₂,S}}
    # Construct the sparse LHS matrix
    problem.A .= constructA!(valuefunction, Δt⁻¹, model, G, calibration, withnegative)

    # Consutruct the RHS
    problem.b .= constructsource(valuefunction, Δt⁻¹, model, G, calibration) - constructadv(valuefunction, model, G, calibration)

    sol = solve!(problem)

    if !SciMLBase.successful_retcode(sol)
        throw("Time step solver failed at time $(valuefunction.t.t)!")
    end

    return problem
end

function updateovergrid!(H, u, θ)
    @inbounds for k in eachindex(u)
        H[k] = θ * u[k] + (1 - θ) * H[k]
    end
end

"Iterate linear solver until convergence"
function steadystate!(valuefunction::ValueFunction{S, N₁, N₂}, Δt::S, model::M, G::GR, calibration; timeiterations = 10_000, picarditerations = 100, printstep = 100, tolerance::Error{S} = Error{S}(1e-3, 1e-3), verbose = 0, withnegative = false, alg = KLUFactorization(), θ = 0.9) where {S, N₁, N₂, M <: UnitIAM{S}, GR <: AbstractGrid{N₁, N₂, S}}
    Δt⁻¹ = 1 / Δt
    A₀ = constructA!(valuefunction, Δt⁻¹, model, G, calibration, withnegative)
    source = constructsource(valuefunction, Δt⁻¹, model, G, calibration)
    b₀ = source - constructadv(valuefunction, model, G, calibration)

    # Initialise the problem
    problem = LinearSolve.init(LinearProblem(A₀, b₀), alg)
    itererror = abserror(problem.u, valuefunction.H)
    
    for iter in 1:timeiterations
        
        constructsource!(source, valuefunction, Δt⁻¹, model, G, calibration)
        
        for k in 1:picarditerations # Stabilise quadratic approximation
            picardstep!(problem, source, valuefunction, Δt⁻¹, model, G, calibration; withnegative)
            updateovergrid!(valuefunction.H, problem.u, θ)
        end
        
        backwardstep!(problem, valuefunction, Δt⁻¹, model, G, calibration; withnegative)

        itererror = abserror(problem.u, valuefunction.H)
        hjberror = maximum(abs, problem.A * vec(valuefunction.H) - problem.b)

        updateovergrid!(valuefunction.H, problem.u, 1.)

        if itererror < tolerance
            return valuefunction, (iter, itererror)
        end

        if (verbose > 1) || (verbose > 0 && iter % printstep == 0)
            @printf "Iteration %d: absolute step = %.2e, relative step = %.2e, HJB error = %.2e\r" iter itererror.absolute itererror.relative hjberror
        end
    end

    @warn @sprintf "Failed convergence in %d iterations.\n" timeiterations

    return valuefunction, (timeiterations, itererror)
end

function richardsonsteadystate!(V₁::ValueFunction{S, N₁, N₂}, Δt::S, model::M, G₁::GR, calibration; verbose = 0, withnegative = false, iterkwargs...) where {S, N₁, N₂, M <: UnitIAM{S}, GR}
    G₂ = halfgrid(G₁)
    V₂ = interpolateovergrid(V₁, G₁, G₂)

    if verbose > 0 println("Solving smaller problem...") end
    steadystate!(V₂, Δt, model, G₂, calibration; verbose, withnegative, iterkwargs...)
    V₂ = interpolateovergrid(V₂, G₂, G₁)

    if verbose > 0 println("Solving larger problem...") end
    V₁.H .= V₂.H
    V₁.α .= V₂.α
    steadystate!(V₁, Δt, model, G₁, calibration; verbose, withnegative, iterkwargs...)

    @inbounds for idx in CartesianIndices(G₁)
        V₁.H[idx] = 2V₁.H[idx] - V₂.H[idx]
        αmax = upperbound(V₁.t.t, G₁[idx], model, calibration, withnegative)
        V₁.α[idx] = clamp(2V₁.α[idx] - V₂.α[idx], 0, αmax)
    end

    return V₁
end

function backwardsimulation!(
    valuefunction::ValueFunction{S, N₁, N₂}, Δt::S, model::M, G::GR, calibration::Calibration; 
    t₀ = zero(S), withnegative = false,
    verbose = 0, printstep = S(10), 
    withsave = true, outdir = "data", overwrite = false, 
    startcache = valuefunction.t.t, cachestep = S(1)) where {S, N₁, N₂, M <: UnitIAM{S}, GR <: AbstractGrid{N₁, N₂, S}}
    
    if withsave
        cachepath, cachefile = initcachefile(model,G, outdir, withnegative; overwrite)
        tcache = copy(startcache)
        magnitude = -floor(Int, log10(abs(cachestep)))
        keyformat = Printf.Format("%.$(magnitude)f")
    end

    if verbose > 0
        tverbose = copy(valuefunction.t.t)
    end

    Δt⁻¹ = 1 / Δt
    n = length(G)
    A₀ = constructA!(valuefunction, Δt⁻¹, model, G, calibration, withnegative)
    b₀ = Vector{S}(undef, n)
    problem = LinearSolve.init(LinearProblem(A₀, b₀), KLUFactorization())

    while t₀ < valuefunction.t.t
        backwardstep!(problem, valuefunction, Δt⁻¹, model, G, calibration; withnegative)
        
        @inbounds for k in eachindex(problem.u)
            valuefunction.H[k] = problem.u[k]
        end

        valuefunction.t.t -= Δt

        if (verbose > 1) || (verbose > 0 && valuefunction.t.t < tverbose)
            if verbose > 0 
                tverbose = tverbose - printstep 
            end
            @printf "Time %.2f\r" valuefunction.t.t
        end

        if withsave && valuefunction.t.t ≤ tcache
            cachekey = Printf.format(keyformat, tcache)
            if verbose > 1 @printf "Saving cache with key %s\n" cachekey end
            
            group = JLD2.Group(cachefile, cachekey)
            group["V"] = valuefunction

            tcache = tcache - cachestep
        end
    end

    if withsave
        close(cachefile)
        if verbose > 0 @printf "Cached in %s" cachepath end
    end

    return valuefunction
end