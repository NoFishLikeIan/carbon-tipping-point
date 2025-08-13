function pointminimisation(u, parameters)
    t, idx, Fₜ₊ₕ, Xᵢ, model, calibration, G = parameters
    F′, Δt = markovstep(t, idx, Fₜ₊ₕ, u.α, model, calibration, G)

    return logcost(F′, t, Xᵢ, Δt, u, model, calibration)
end

function backwardstep!(state::DPState, cluster, Δts, model, calibration, G; iterkwargs...)
    backwardstep!(state.valuefunction, state.policystate, state.timestate, cluster, Δts, model, calibration, G; iterkwargs...)
end

function inverseidentity(::Policy{T}) where T <: Real
    MMatrix{2, 2, T}(1, 0, 0, 1)
end

function backwardstep!(valuefunction::ValueFunction{T}, policystate::PolicyState{T}, timestate::Time{T}, cluster, Δts, model, calibration, G; withnegative = true, ad = Optimization.AutoForwardDiff(), prevweight = 0.5, lb = Policy(0.1, 0.), optkwargs...) where T

    fn = OptimizationFunction(pointminimisation, ad)
    indices = CartesianIndices(G)

    @inbounds @threads for (i, δt) in cluster
        idx = indices[i]
        Xᵢ = G.X[idx]
        t = timestate.τ - δt
        pₜ = policystate.policy[idx]
        
        parameters = (t, idx, valuefunction.Fₜ₊ₕ, Xᵢ, model, calibration, G)
        
        ᾱ = withnegative ? one(T) : γ(t, calibration) + δₘ(Xᵢ.M, model.hogg)

        if ᾱ ≥ 0
            ub = Policy(0.9, ᾱ)

            α₀ = prevweight * clamp(pₜ.α, 0, ᾱ) + (1 - prevweight) * ᾱ / 2
            u₀ = Policy(pₜ.χ, α₀)

            prob = Optimization.OptimizationProblem(fn, u₀, parameters; lb = lb, ub = ub)

            sol = solve(prob, BFGS(initial_invH = inverseidentity); optkwargs...)

            pₜ .= sol.u
            policystate.foc[idx] = sol.original.g_residual
            valuefunction.Fₜ[idx] = exp(sol.objective)
            Δts[i] = timestep(t, Xᵢ, sol.u.α, model, calibration, G)
            timestate.t[idx] = t
        else
            obj = @closure χ -> pointminimisation(Policy(χ, ᾱ), parameters)
            y, χ = gssmin(obj, 0.1, 0.9)
            
            pₜ.χ = χ
            pₜ.α = ᾱ
            policystate.foc[idx] = zero(χ)

            valuefunction.Fₜ[idx] = exp(y)
            Δts[i] = timestep(t, Xᵢ, ᾱ, model, calibration, G)
            timestate.t[idx] = t
        end
    end

    valuefunction.Fₜ₊ₕ .= valuefunction.Fₜ

    return nothing
end

function backwardsimulation!(state, model, calibration, G; kwargs...)
    queue = DiagonalRedBlackQueue(G)
    backwardsimulation!(queue, state, model, calibration, G; kwargs...)
end

function backwardsimulation!(queue::ZigZagBoomerang.PartialQueue, state, model, calibration, G; verbose = 0, cachepath = nothing, cachestep = 0.25, overwrite = false, tstop = 0., withnegative = false, tcacheinit = state.timestate.τ, printevery = 10_000, iterkwargs...)
    tcache = tcacheinit
    inittime = time()

    savecache = !isnothing(cachepath)
    if savecache
        if isfile(cachepath) && overwrite
            if (verbose ≥ 1) 
                @warn "File $cachepath already exists and mode is overwrite. Will remove."
            end

            rm(cachepath)

            cachefile = jldopen(cachepath, "w+")
            cachefile["G"] = G
            cachefile["model"] = model

        elseif isfile(cachepath) && !overwrite 
            if (verbose ≥ 1)
                println("File $cachepath already exists and mode is not overwrite. Will resume from cache.")
            end

            cachefile = jldopen(cachepath, "a+")
            tcache = tcacheinit - minimum(queue.vals) - cachestep
        else
            cachefile = jldopen(cachepath, "w+")
            cachefile["G"] = G
            cachefile["model"] = model
        end
    end

    Δts = Vector{Float64}(undef, prod(size(G)))
    passcounter = 1

    while !isempty(queue)
        tmin = tcacheinit - minimum(queue.vals)
        cluster = ZigZagBoomerang.dequeue!(queue) |> only
        
        if (verbose ≥ 2) || ((verbose ≥ 1) && (passcounter % printevery == 0))
            elapsed = time() - inittime
            @printf "%.2f: pass %i, cluster (size %i) minimum time = %.4f\n" elapsed passcounter length(cluster) tmin
            flush(stdout)
        end

        backwardstep!(state, cluster, Δts, model, calibration, G; withnegative = withnegative, iterkwargs...)

        passcounter += 1
        
        for (i, _) in cluster
            if queue[i] ≤ tcacheinit - tstop
                queue[i] += Δts[i]
            end
        end
        
        if savecache && tmin ≤ tcache
            if (verbose ≥ 2)
                println("Saving cache at $tcache")
            end

            group = JLD2.Group(cachefile, "$tcache")
            group["state"] = state
            tcache = tcache - cachestep 
        end
    end

    if savecache 
        close(cachefile)
        if (verbose ≥ 1)
            println("$(now()): ", "Saved cached file into $cachepath")
        end
    end
end

function computebackward(model, calibration, G; outdir = "data", kwargs...)
    computebackward(loadterminal(model; outdir), model, calibration, G; outdir, kwargs...)
end
function computebackward(terminalresults, model, calibration, G; verbose = 0, withsave = true, outdir = "data", withnegative = false, iterkwargs...)
    terminalstate, terminalG = terminalresults
    state = interpolateovergrid(terminalstate, terminalG, G)

    if withsave
        folder = simpaths(model)
        cachefolder = joinpath(outdir, folder)
        if !isdir(cachefolder) mkpath(cachefolder) end
        
        filename = makefilename(model)
    end

    cachepath = withsave ? joinpath(cachefolder, filename) : nothing

    backwardsimulation!(state, model, calibration, G; verbose, cachepath, withnegative, iterkwargs...)

    return state
end
