function updateproblem!(problem, valuefunction::ValueFunction, Δt⁻¹, model::M, G::RegularGrid{N₁,N₂,S}, calibration::Calibration; withnegative = false) where {N₁, N₂, S, M <: UnitElasticityModel{S}}
    problem.A .= constructA(valuefunction, Δt⁻¹, model, G, calibration; withnegative)
    constructb!(problem.b, valuefunction, Δt⁻¹, model, G, calibration)

    return problem
end

function backwardstep!(problem, valuefunction::ValueFunction, Δt⁻¹, model::M, G::RegularGrid{N₁,N₂,S}, calibration::Calibration; withnegative = false) where {N₁, N₂, S, M <: UnitElasticityModel{S}}
    updateproblem!(problem, valuefunction, Δt⁻¹, model, G, calibration; withnegative)
    sol = solve!(problem)

    if !SciMLBase.successful_retcode(sol)
        throw("Linear solver failed at time $(valuefunction.t.t)!")
    end

    return problem
end

"Iterate linear solver until convergence"
function steadystate!(valuefunction::ValueFunction{S, N₁, N₂}, Δt::S, model::M, G, calibration; iterations = 10_000, printstep = iterations ÷ 100, tolerance::Error{S} = Error{S}(1e-3, 1e-3), verbose = 0, withnegative = false) where {S, N₁, N₂, M <: UnitElasticityModel{S}}
    Δt⁻¹ = 1 / Δt
    A₀ = constructA(valuefunction, Δt⁻¹, model, G, calibration)
    b₀ = constructb(valuefunction, Δt⁻¹, model, G, calibration)

    problem = LinearSolve.init(LinearProblem(A₀, b₀), KLUFactorization())

    for iter in 1:iterations
        backwardstep!(problem, valuefunction, Δt⁻¹, model, G, calibration; withnegative)

        itererror = abserror(problem.u, valuefunction.H)
        valuefunction.H .= reshape(problem.u, size(G))

        if itererror < tolerance
            return valuefunction, (iter, itererror)
        end

        if (verbose > 1) || (verbose > 0 && iter % printstep == 0)
            @printf "Iteration %d: absolute error = %.2e, relative error = %.2e\r" iter itererror.absolute itererror.relative
        end
    end

    @warn @sprintf "Failed convergence in %d iterations.\n" iterations

    return valuefunction, (iterations, Error{S}(NaN, NaN))
end

function backwardsimulation!(
    valuefunction::ValueFunction{S, N₁, N₂}, Δt::S, model::M, G::RegularGrid{N₁, N₂, S}, calibration::Calibration; 
    t₀ = zero(S), withnegative = false,
    verbose = 0, printstep = S(10), 
    withsave = true, outdir = "data", overwrite = false, 
    startcache = valuefunction.t.t, cachestep = S(1)) where {S, N₁, N₂, M <: UnitElasticityModel{S}}
    
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
    A₀ = constructA(valuefunction, Δt⁻¹, model, G, calibration)
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
            
            centralpolicy!(valuefunction, model, G, calibration) # Recomputes α via central difference for smoothness
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