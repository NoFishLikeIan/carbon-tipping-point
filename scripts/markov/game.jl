include("backward.jl") # Extends backward scripts to game model

Values = Vector{NTuple{2, Matrix{Float64}}}
Policies = Vector{Array{Float64, 3}}

function backwardstep!(Δts, Fs::Values, policies::Policies, cluster, models::Vector{<:AbstractModel}, regionalcalibrations::Vector{Calibration}, calibration::Calibration, G; αfactor = 1.5, allownegative = false,optargs...)

    for (i, δt) in cluster
        n = length(models) # Number of players
        indices = CartesianIndices(G)

        idx = indices[i]
        t = models[1].economy.τ - δt
        M = exp(G.X[idx].m)
        Δtᵢ = Inf
        
        for p in 1:n
            Fₜ, Fₜ₊ₕ = Fs[p]
            policy = policies[p]
            model = models[p]

            ᾱ = γ(t, regionalcalibrations[p]) + δₘ(M, model.hogg)
            α₋ᵢ = sum(policies[j][idx, 2] for j in 1:n if j ≠ p)
            u₀ = copy(policy[idx, :])

            objective = @closure u -> begin
                Fᵉₜ, Δt = markovstep(t, idx, Fₜ₊ₕ, u[2], α₋ᵢ, model, calibration, G) # Transition uses the global calibration
                
                return logcost(Fᵉₜ, t, G.X[idx], Δt, u, model, regionalcalibrations[p]) # Costs use the regional calibration
            end

            diffobjective = TwiceDifferentiable(objective, u₀; autodiff = :forward)

            # Solve first unconstrained problem
            openinterval = TwiceDifferentiableConstraints([0., 0.], [1., 2ᾱ])
            u₀[2] = αfactor * ᾱ
            optimum = similar(u₀)

            fallbackoptimisation!(optimum, diffobjective, u₀, idx, t, policy, openinterval, G; optargs...)

            if !allownegative # Solve constrained problem with χ from unconstrained problem
                u₀[1] = optimum[1]
                u₀[2] = min(optimum[2], ᾱ / 2) # Guarantees u₀ ∈ U
                constraints = TwiceDifferentiableConstraints([0., 0.], [1., ᾱ])

                fallbackoptimisation!(optimum, diffobjective, u₀, idx, t, policy, constraints, G; optargs...)
            end

            objectiveminimum = objective(optimum)

            policy[idx, :] .= optimum
            Fₜ[idx] = exp(objectiveminimum)
            Δtᵢ = min(last(markovstep(t, idx, Fₜ₊ₕ, policy[idx, 2], model, calibration, G)), Δtᵢ)
        end

        Δts[i] = Δtᵢ
    end
end

function backwardsimulation!(Fs::Values, policies::Policies, models::Vector{<:AbstractModel}, regionalcalibrations::Vector{Calibration}, calibration::Calibration, G; kwargs...)
    queue = DiagonalRedBlackQueue(G)
    backwardsimulation!(queue, Fs, policies, models, regionalcalibrations, calibration, G; kwargs...)
end

function backwardsimulation!(queue::PartialQueue, Fs::Values, policies::Policies, models::Vector{<:AbstractModel}, regionalcalibrations::Vector{Calibration}, calibration::Calibration, G; verbose = 0, cachepath = nothing, cachestep = 0.25, overwrite = false, tstop = 0., tcache = models[1].economy.τ, stepkwargs...)
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
            cachefile["models"] = models

        elseif isfile(cachepath) && !overwrite 

            if (verbose ≥ 1)
                println("File $cachepath already exists and mode is not overwrite. Will resume from cache.")
            end

            cachefile = jldopen(cachepath, "a+")
            tcache = models[1].economy.τ - minimum(queue.vals) - cachestep

        else
            
            cachefile = jldopen(cachepath, "w+")
            cachefile["G"] = G
            cachefile["models"] = models

        end
    end

    Δts = Vector{Float64}(undef, prod(size(G)))
    passcounter = 1

    while !isempty(queue)
        tmin = models[1].economy.τ - minimum(queue.vals)

        if (verbose ≥ 2) || ((verbose ≥ 1) && (passcounter % 500 == 0))
            @printf("%s: pass %i, cluster minimum time = %.4f\n", now(), passcounter, tmin)
            flush(stdout)
        end

        passcounter += 1
        
        clusters = dequeue!(queue)
        for cluster in clusters
            backwardstep!(Δts, Fs, policies, cluster, models, regionalcalibrations, calibration, G; verbose, stepkwargs...)

            indices = first.(cluster)

            for i in indices
                if queue[i] ≤ models[1].economy.τ - tstop
                    queue[i] += Δts[i]
                end
            end
        end
        
        if savecache && tmin ≤ tcache
            if (verbose ≥ 2)
                println("Saving cache at $tcache")
            end

            group = JLD2.Group(cachefile, "$tcache")
            group["Fs"] = Fs
            group["policies"] = policies
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

function computebackward(models::Vector{<:AbstractModel}, regionalcalibrations::Vector{Calibration}, calibration::Calibration, G::RegularGrid;  outdir = "data", addpaths = repeat([""], length(models)), kwargs...)
    terminalresults = loadterminal(models; outdir, addpaths)
    computebackward(terminalresults, models, regionalcalibrations, calibration, G; outdir, kwargs...)
end
function computebackward(terminalresults, models::Vector{<:AbstractModel}, regionalcalibrations::Vector{Calibration}, calibration::Calibration, G::RegularGrid; verbose = 0, withsave = true, outdir = "data", iterkwargs...)

    dims = length(size(G))
    policies = Array{Float64, dims + 1}[]
    Fs = NTuple{2, Array{Float64, dims}}[]
    
    for (i, terminalresult) in enumerate(terminalresults)
        F̄, terminalconsumption, terminalG = terminalresult
        Fₜ₊ₕ = interpolateovergrid(terminalG, G, F̄)
        Fₜ = similar(Fₜ₊ₕ)
        F = (Fₜ, Fₜ₊ₕ)
    
        policy = Array{Float64}(undef, size(G)..., 2)
        policy[:, :, 1] .= interpolateovergrid(terminalG, G, terminalconsumption)
        policy[:, :, 2] .= γ(models[i].economy.τ, regionalcalibrations[i])
    
        push!(policies, policy)
        push!(Fs, F)
    end

    if withsave
        if !isdir(outdir) mkpath(outdir) end
        
        filename = makefilename(models)
    end

    cachepath = withsave ? joinpath(outdir, filename) : nothing
    
    backwardsimulation!(Fs, policies, models, regionalcalibrations, calibration, G; verbose, cachepath, iterkwargs...)
end