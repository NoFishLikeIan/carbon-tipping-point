function logminimisation(u, parameters)
    t, idx, Fₜ₊ₕ, Xᵢ, Δtmax, model, calibration, G = parameters
    
    F′, Δt = markovstep(t, idx, Fₜ₊ₕ, u, Δtmax, model, calibration, G)

    return logcost(F′, Δt, t, Xᵢ, u, model)
end

function inverseidentity(::Policy{T}) where T <: Real
    MMatrix{2, 2, T}(1, 0, 0, 1)
end

function backwardstep!(state::DPState, cluster, Δts, model, calibration, G; iterkwargs...)
    backwardstep!(state.valuefunction, state.policystate, state.timestate, cluster, Δts, model, calibration, G; iterkwargs...)
end
function backwardstep!(valuefunction::ValueFunction{T}, policystate::PolicyState{T}, timestate::Time{T}, cluster, Δts, model::M, calibration, G; 
    withnegative = true, ad = Optimization.AutoForwardDiff(), alg = BFGS(initial_invH = inverseidentity, linesearch = BackTracking(order = 3)), Δtmax = 1 / 100, optkwargs...) where {T, D <: Damages{T}, P <: LogSeparable{T}, M <: AbstractModel{T, D, P}}

    fn = OptimizationFunction(logminimisation, ad)
    indices = CartesianIndices(G)
    lb = Policy{T}(0., 0.); lbs = lb .+ 1e-3;
    ub = Policy{T}(1., ifelse(withnegative, Inf, 1.)); ubs = ub .- 1e-3;

    @inbounds @threads for (i, δt) in cluster
        idx = indices[i]
        Xᵢ = G.X[idx]
        t = timestate.τ - δt

        parameters = (t, idx, valuefunction.Fₜ₊ₕ, Xᵢ, Δtmax, model, calibration, G)

        u₀ = clamp.(policystate.policy[idx], lbs, ubs)
        prob = Optimization.OptimizationProblem(fn, u₀, parameters; lb = lb, ub = ub)
        sol = solve(prob, alg; optkwargs...)

        if !SciMLBase.successful_retcode(sol)
            @warn "Optimization failed for idx=$idx at time t=$t with retcode $(sol.retcode)"
        end

        policystate.policy[idx] .= sol.u
        valuefunction.Fₜ[idx] = sol.objective
        policystate.foc[idx] = sol.original.g_residual
        Δts[i] = timestep(t, Xᵢ, sol.u, Δtmax, model, calibration, G)
        timestate.t[idx] = t
    end

    valuefunction.Fₜ₊ₕ .= valuefunction.Fₜ

    return nothing
end

function backwardsimulation!(state, model, calibration, G; kwargs...)
    δt₀ = vec(state.timestate.τ .- state.timestate.t)
    queue = DiagonalRedBlackQueue(G; initialvector = δt₀)
    backwardsimulation!(queue, state, model, calibration, G; kwargs...)
end

function backwardsimulation!(queue::ZigZagBoomerang.PartialQueue, state, model, calibration, G; verbose = 0, cachepath = nothing, cachestep = 0.25, overwrite = false, tstop = 0., withnegative = false, tcacheinit = state.timestate.τ, printevery = 10_000, iterkwargs...)
    clustertime = state.timestate.τ - minimum(queue.vals)
    tcache = min(tcacheinit, maximum(state.timestate.t)) # Next time at which to save a cache.
    if verbose ≥ 1
        inittime = time()
        elapsed = 0.0
    end

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
            tcache =  min(tcacheinit, maximum(state.timestate.t)) - minimum(queue.vals) - cachestep
        else
            cachefile = jldopen(cachepath, "w+")
            cachefile["G"] = G
            cachefile["model"] = model
        end
    end

    Δts = Vector{Float64}(undef, prod(size(G)))
    passcounter = 1

    while !isempty(queue)
        cluster = ZigZagBoomerang.dequeue!(queue) |> only
        clustertime = state.timestate.τ - minimum(queue.vals)
        
        if (verbose ≥ 2) || ((verbose ≥ 1) && (passcounter % printevery == 0))
            nextelapsed = printbackward(elapsed, inittime, passcounter, cluster, clustertime)
            elapsed = nextelapsed
        end

        backwardstep!(state, cluster, Δts, model, calibration, G; withnegative = withnegative, iterkwargs...)

        passcounter += 1
        
        for (i, _) in cluster
            if state.timestate.τ - queue[i] ≥ tstop
                queue[i] += Δts[i]
            end
        end
        
        if savecache && (state.timestate.τ - minimum(queue.vals)) ≤ tcache
            if (verbose ≥ 2) println("Saving cache with key $tcache") end

            group = JLD2.Group(cachefile, "$tcache")
            group["state"] = state

            tcache = tcache - cachestep # Update next time at which to cache
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
