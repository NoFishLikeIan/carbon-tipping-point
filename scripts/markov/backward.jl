function pointminimisation(u, parameters)
    t, idx, Fₜ₊ₕ, Xᵢ, model, calibration, G = parameters
    Fᵉₜ, Δt = markovstep(t, idx, Fₜ₊ₕ, u.α, model, calibration, G)

    return cost(Fᵉₜ, t, Xᵢ, Δt, u, model, calibration)
end

function backwardstep!(F, policy, cluster, foc, Δts, model, calibration, G; withnegative = true, ad = Optimization.AutoForwardDiff(), itpε = 0.5, optkwargs...)

    Fₜ, Fₜ₊ₕ = F
    fn = OptimizationFunction(pointminimisation, ad)
    indices = CartesianIndices(G)

    @inline @threads for (i, δt) in cluster
        idx = indices[i]
        Xᵢ = G.X[idx]
        t = model.economy.τ - δt
        parameters = (t, idx, Fₜ₊ₕ, Xᵢ, model, calibration, G)
        
        ᾱ = withnegative ? 2.0 : γ(t, calibration) + δₘ(Xᵢ.M, model.hogg)

        if ᾱ ≥ 0
            lb = Policy(0.1, 0.)
            ub = Policy(0.9, ᾱ)

            α₀ = (1 - itpε) * clamp(policy[idx].α, 0, ᾱ) + itpε * ᾱ / 2
            u₀ = Policy(0.5, α₀)

            prob = Optimization.OptimizationProblem(fn, u₀, parameters; lb = lb, ub = ub)

            sol = solve(prob, LBFGS(); iterations = 10_000, time_limit = 0.5, optkwargs...)

            policy[idx] .= sol.u
            foc[idx] = sol.original.g_residual
            Fₜ[idx] = sol.objective
            Δts[i] = timestep(t, Xᵢ, sol.u.α, model, calibration, G)
        else
            obj = @closure χ -> pointminimisation(Policy(χ, ᾱ), parameters)
            y, χ = gssmin(obj, 0.1, 0.9)
            
            policy[idx][1] = χ
            policy[idx][2] = ᾱ
            foc[idx] = zero(χ)
            Fₜ[idx] = y
            Δts[i] = timestep(t, Xᵢ, ᾱ, model, calibration, G)
        end
    end
end

function backwardsimulation!(F, policy, foc, model, calibration, G; kwargs...)
    queue = DiagonalRedBlackQueue(G)
    backwardsimulation!(queue, F, policy, foc, model, calibration, G; kwargs...)
end

function backwardsimulation!(queue::ZigZagBoomerang.PartialQueue, F, policy, foc, model, calibration, G; verbose = 0, cachepath = nothing, cachestep = 0.25, overwrite = false, tstop = 0., withnegative = false, tcacheinit = model.economy.τ)
    tcache = tcacheinit # Just to make sure it is well defined in all paths.

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

        if (verbose ≥ 2) || ((verbose ≥ 1) && (passcounter % 1_000 == 0))
            @printf("%s: pass %i, cluster minimum time = %.4f\n", now(), passcounter, tmin)
            flush(stdout)
        end

        passcounter += 1
        
        clusters = ZigZagBoomerang.dequeue!(queue)
        for cluster in clusters
            backwardstep!(F, policy, cluster, foc, Δts, model, calibration, G; withnegative)

            indices = first.(cluster)

            for i in indices
                if queue[i] ≤ tcacheinit - tstop
                    queue[i] += Δts[i]
                end
            end
        end
        
        if savecache && tmin ≤ tcache
            if (verbose ≥ 2)
                println("Saving cache at $tcache")
            end

            group = JLD2.Group(cachefile, "$tcache")
            group["F"] = first(F)
            group["policy"] = policy
            group["foc"] = foc
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
    terminalresults = loadterminal(model; outdir)
    computebackward(terminalresults, model, calibration, G; outdir, kwargs...)
end
function computebackward(terminalresults, model, calibration, G; verbose = 0, withsave = true, outdir = "data", withnegative = false, iterkwargs...)
    F̄, terminalconsumption, terminalG = terminalresults
    Fₜ₊ₕ = interpolateovergrid(F̄, terminalG, G)
    Fₜ = copy(Fₜ₊ₕ)
    F = (Fₜ, Fₜ₊ₕ)
    
    ᾱ = max(γ(calibration.τ, calibration), 0)
    terminalpolicy = [Policy(terminalconsumption[idx], ᾱ) for idx in CartesianIndices(terminalG)]

    policy = interpolateovergrid(terminalpolicy, terminalG, G)
	foc = fill(Inf, size(G))

    if withsave
        folder = simpaths(model)
        cachefolder = joinpath(outdir, folder)
        if !isdir(cachefolder) mkpath(cachefolder) end
        
        filename = makefilename(model)
    end

    cachepath = withsave ? joinpath(cachefolder, filename) : nothing
    backwardsimulation!(F, policy, foc, model, calibration, G; verbose, cachepath, withnegative, iterkwargs...)

    return F, policy, foc
end
