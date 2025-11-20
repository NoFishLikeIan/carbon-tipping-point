"Updates stencils, source and policy and takes a step from `H(t)` to `H(t - Δt)`, under `ϵ → 0`"
function backwardequilibriumstep!(problem, Sᵨ, equilibriumstencil, (t, H, α), Δt⁻¹, model::M, G::GR, calibration::Calibration; withnegative = true) where {N₁, N₂, S, D, P, C <: LinearClimate, M <: UnitIAM{S, D, P, C}, GR <: AbstractGrid{N₁, N₂, S}}
    # Construct the sparse LHS matrix
    constructequilibriumDᵐ!(equilibriumstencil, (t, H, α), model, G, calibration, withnegative)
    problem.A = Sᵨ - sparse(equilibriumstencil[1], equilibriumstencil[2], equilibriumstencil[3], N₂, N₂)
    
    # Consutruct the RHS
    constructequilibriumsource!(problem.b, (t, H, α), Δt⁻¹, model, G, calibration)
    sol = solve!(problem)

    if !SciMLBase.successful_retcode(sol)
        throw("Time step solver failed at time $(valuefunction.t.t)!")
    end
end

"Updates stencils, source and policy and takes a step from `H(t)` to `H(t - Δt)`."
function backwardstep!(problem, R, stencilm, valuefunction::ValueFunction, Δt⁻¹, model::M, G::GR, calibration::Calibration; withnegative = true) where {N₁, N₂, S, M <: UnitIAM{S}, GR <: AbstractGrid{N₁,N₂,S}}
    # Construct the sparse LHS matrix
    n = length(G)
    constructDᵐ!(stencilm, valuefunction, model, G, calibration, withnegative)
    problem.A = R - sparse(stencilm[1], stencilm[2], stencilm[3], n, n)
    
    # Consutruct the RHS
    constructsource!(problem.b, valuefunction, Δt⁻¹, model, G, calibration)
    sol = solve!(problem)

    if !SciMLBase.successful_retcode(sol)
        throw("Time step solver failed at time $(valuefunction.t.t)!")
    end

    return sol
end

function equilibriumsteadystate!(valuefunction::ValueFunction{S, N₁, N₂}, Δt::S, model::M, G::RegularGrid{N₁, N₂, S}, calibration; timeiterations = 10_000, printstep = 100, tolerance::Error{S} = Error{S}(1e-6, 1e-4), verbose = 0, withnegative = true, alg = KLUFactorization()) where {N₁, N₂, S, D, P, C <: LinearClimate, M <: UnitIAM{S, D, P, C}}
    Δt⁻¹ = 1 / Δt

    Sᵨ = (preferences.ρ + Δt⁻¹) * I

    linearmodel = linearIAM(model)
    t = valuefunction.t.t
    H = @view valuefunction.H[1, :]
    α = @view valuefunction.α[1, :] # Compute on a given row and copy once convergence is done 
    
    equilibriumstencil = makeequilibriumstencil(G)
    constructequilibriumDᵐ!(equilibriumstencil, (t, H, α), linearmodel, G, calibration, withnegative)
    beq = constructequilibriumsource((t, H, α), Δt⁻¹, linearmodel, G, calibration)
    Aeq = Sᵨ - sparse(equilibriumstencil[1], equilibriumstencil[2], equilibriumstencil[3], N₂, N₂)
    problem = LinearSolve.init(LinearProblem(Aeq, beq), alg)
    backwardequilibriumstep!(problem, Sᵨ, equilibriumstencil, (t, H, α), Δt⁻¹, linearmodel, G, calibration; withnegative)

    itererror = abserror(problem.u, H)
    if itererror < tolerance
        @inbounds for i in 2:N₁
            valuefunction.α[i, :] .= α
            valuefunction.H[i, :] .= H
        end

        return valuefunction, (1, itererror)
    end
    
    for iter in 2:timeiterations  
        backwardequilibriumstep!(problem, Sᵨ, equilibriumstencil, (t, H, α), Δt⁻¹, linearmodel, G, calibration; withnegative)
        itererror = abserror(problem.u, H)

        H .= problem.u

        if itererror < tolerance
            @inbounds for i in 2:N₁
                valuefunction.α[i, :] .= α
                valuefunction.H[i, :] .= H
            end
            
            return valuefunction, (iter, itererror)
        end

        if (verbose > 1) || (verbose > 0 && iter % printstep == 0)
            @printf "Iteration %d: absolute step = %.2e, relative step = %.2e\r" iter itererror.absolute itererror.relative
        end
    end

    @inbounds for i in 2:N₁
        valuefunction.α[i, :] .= α
        valuefunction.H[i, :] .= H
    end

    return valuefunction, (timeiterations, itererror)
end

"Iterate linear solver until convergence"
function steadystate!(valuefunction::ValueFunction{S, N₁, N₂}, Δt::S, model::M, G::RegularGrid{N₁, N₂, S}, calibration; timeiterations = 10_000, printstep = 100, tolerance::Error{S} = Error{S}(1e-6, 1e-4), verbose = 0, withnegative = true, alg = KLUFactorization()) where {S, N₁, N₂, M <: UnitIAM{S}}    
    # Initialise problem
    Δt⁻¹ = 1 / Δt
    n = length(G)
    stencilT, stencilm = makestencil(G)
    constructDᵀ!(stencilT, model, G)
    constructDᵐ!(stencilm, valuefunction, model, G, calibration, withnegative)
    b₀ = constructsource(valuefunction, Δt⁻¹, model, G, calibration)
    Sᵨ = (preferences.ρ + Δt⁻¹) * I
    R = Sᵨ - sparse(stencilT[1], stencilT[2], stencilT[3], n, n)
    A₀ = R - sparse(stencilm[1], stencilm[2], stencilm[3], n, n)
    problem = LinearSolve.init(LinearProblem(A₀, b₀), alg)
    
    # First iteration
    backwardstep!(problem, R, stencilm, valuefunction, Δt⁻¹, model, G, calibration; withnegative)
    itererror = abserror(problem.u, valuefunction.H)
    if itererror < tolerance return valuefunction, (1, itererror) end
    
    for iter in 2:timeiterations  
        backwardstep!(problem, R, stencilm, valuefunction, Δt⁻¹, model, G, calibration; withnegative)
        itererror = abserror(problem.u, valuefunction.H)

        @inbounds for (k, uₖ) in enumerate(problem.u)
            valuefunction.H[k] = uₖ
        end

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

function inwardoutwardsteadystate!(valuefunction::ValueFunction{S, N₁, N₂}, Δt::S, model::M, G::RegularGrid{N₁, N₂, S}, calibration; equilibriumtolerance = Error{S}(1e-8, 1e-8), tolerance = Error{S}(1e-6, 1e-6), verbose = 0, kwargs...) where {S, N₁, N₂, M <: UnitIAM{S}}
    # Start by solving the ϵ → 0 problem and using as initial guess
    if verbose > 0
        @printf "Solving ϵ → 0 problem\n"
    end
    
    equilibriumsteadystate!(valuefunction, Δt, model, G, calibration; tolerance = equilibriumtolerance, verbose, kwargs...)

    if verbose > 0
        @printf "\nSolving ϵ = %.2f problem\n" model.climate.hogg.ϵ
    end

    steadystate!(valuefunction, Δt, model, G, calibration; tolerance, verbose, kwargs...)

end

"Backward simulation of `valuefunction` from `valuefunction.t` to `t₀`. Returns a `OrderedDict` with either starting and terminal `valuefunction`, if `storetrajectory` is `false`, or the whole trajectory, otherwise."
function backwardsimulation!(valuefunction::ValueFunction{S, N₁, N₂}, Δt::S, model::M, G::GR, calibration::Calibration; t₀ = zero(S), withnegative = true, verbose = 0, printstep = 10, withsave = true, outdir = "data", overwrite = false, startcache = valuefunction.t.t, cachestep = S(1), alg = KLUFactorization(), storetrajectory = false) where {S, N₁, N₂, M <: UnitIAM{S}, GR <: AbstractGrid{N₁, N₂, S}}
    
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
    constructDᵐ!(stencilm, valuefunction, model, G, calibration, withnegative)
    b₀ = constructsource(valuefunction, Δt⁻¹, model, G, calibration)
    Sᵨ = (preferences.ρ + Δt⁻¹) * I
    R = Sᵨ - sparse(stencilT[1], stencilT[2], stencilT[3], n, n)
    A₀ = R - sparse(stencilm[1], stencilm[2], stencilm[3], n, n)
    problem = LinearSolve.init(LinearProblem(A₀, b₀), alg)
    
    # First iteration
    backwardstep!(problem, R, stencilm, valuefunction, Δt⁻¹, model, G, calibration; withnegative)
 
    while t₀ < valuefunction.t.t
        backwardstep!(problem, R, stencilm, valuefunction, Δt⁻¹, model, G, calibration; withnegative)

        valuefunction.t.t -= Δt

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