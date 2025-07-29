function backwardstep!(Δts, F::NTuple{2, Matrix}, policy, cluster, model::AbstractModel, calibration::Calibration, G; withnegative = true)
    Fₜ, Fₜ₊ₕ = F

    @inbounds @threads for (i, δt) in cluster
        indices = CartesianIndices(G)
        
        idx = indices[i]
        Xᵢ = G.X[idx]
        t = model.economy.τ - δt
        M = exp(Xᵢ.m) * hogg.Mᵖ

        objective = @closure u -> begin
            Fᵉₜ, Δt = markovstep(t, idx, Fₜ₊ₕ, u[2], model, calibration, G)
            return cost(Fᵉₜ, t, Xᵢ, Δt, u, model, calibration)
        end

        ᾱ = withnegative ? Inf : γ(t, calibration) + δₘ(M, model.hogg)

        lb = MVector{2}(0., 0.)
        ub = MVector{2}(1., ᾱ)

        if !withnegative # Ensure α₀ < ᾱ
            policy[idx][2] = ᾱ / 2
        end
        
        result = Optim.optimize(objective, lb, ub, policy[idx], Fminbox(NelderMead()))

        policy[idx] .= result.minimizer
        Fₜ[idx] = result.minimum
        Δts[i] = timestep(t, Xᵢ, result.minimizer[2], model, calibration, G)
    end 
end

function backwardsimulation!(F::NTuple{2, Matrix{Float64}}, policy, model::AbstractModel, calibration::Calibration, G; kwargs...)
    queue = DiagonalRedBlackQueue(G)
    backwardsimulation!(queue, F, policy, model, calibration, G; kwargs...)
end

function backwardsimulation!(queue::ZigZagBoomerang.PartialQueue, F::NTuple{2, Matrix{Float64}}, policy, model::AbstractModel, calibration::Calibration, G; verbose = 0, cachepath = nothing, cachestep = 0.25, overwrite = false, tstop = 0., tcache = model.economy.τ, withnegative = false)
    tcache = tcache # Just to make sure it is well defined in all paths.

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
            tcache = model.economy.τ - minimum(queue.vals) - cachestep
        else
            
            cachefile = jldopen(cachepath, "w+")
            cachefile["G"] = G
            cachefile["model"] = model
        end
    end

    Δts = Vector{Float64}(undef, prod(size(G)))
    passcounter = 1

    while !isempty(queue)
        tmin = model.economy.τ - minimum(queue.vals)

        if (verbose ≥ 2) || ((verbose ≥ 1) && (passcounter % 500 == 0))
            @printf("%s: pass %i, cluster minimum time = %.4f\n", now(), passcounter, tmin)
            flush(stdout)
        end

        passcounter += 1
        
        clusters = ZigZagBoomerang.dequeue!(queue)
        for cluster in clusters
            backwardstep!(Δts, F, policy, cluster, model, calibration, G; withnegative)

            indices = first.(cluster)

            for i in indices
                if queue[i] ≤ model.economy.τ - tstop
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

function computebackward(model::AbstractModel, calibration::Calibration, G; outdir = "data", kwargs...)
    terminalresults = loadterminal(model; outdir)
    computebackward(terminalresults, model, calibration, G; outdir, kwargs...)
end
function computebackward(terminalresults, model::AbstractModel, calibration::Calibration, G; verbose = 0, withsave = true, outdir = "data", withnegative = false, iterkwargs...)
    F̄, terminalconsumption, terminalG = terminalresults
    Fₜ₊ₕ = interpolateovergrid(terminalG, G, F̄);
    Fₜ = similar(Fₜ₊ₕ)
    F = (Fₜ, Fₜ₊ₕ)

    policy = Array{Float64}(undef, size(G, 1), size(G, 2), 2)
    policy[:, :, 1] .= interpolateovergrid(terminalG, G, terminalconsumption)
    policy[:, :, 2] .= γ(model.economy.τ, calibration)

    if withsave
        folder = SIMPATHS[typeof(model)]
        cachefolder = joinpath(outdir, folder)
        if !isdir(cachefolder) mkpath(cachefolder) end
        
        filename = makefilename(model)
    end

    cachepath = withsave ? joinpath(cachefolder, filename) : nothing

    backwardsimulation!(F, policy, model, calibration, G; verbose, cachepath, withnegative, iterkwargs...)

    return F, policy
end
