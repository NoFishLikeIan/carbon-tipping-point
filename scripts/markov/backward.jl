using JLD2
using Printf: @printf

using Model, Grid
using FastClosures: @closure
using Printf: @printf
using ZigZagBoomerang: dequeue!, PartialQueue
using Base.Threads: @threads
using SciMLBase: successful_retcode
using Optim
using Statistics: mean

include("chain.jl")

const defaultoptoptions = Optim.Options(g_tol = 1e-12, iterations = 100_000)

"""
Optimise `diffobjective` and stores the minimiser in u. If the optimisation does not converge, it takes the mean of the policy in the neighbourhood.
"""
function fallbackoptimisation!(u,
    diffobjective, u₀, 
    idx, t, policy, 
    constraints::TwiceDifferentiableConstraints, G::RegularGrid; 
    optoptions = defaultoptoptions, verbose = 0)

    if !Optim.isinterior(constraints, u₀)
        throw(ArgumentError("Initial guess is not in the interior of the constraints at t = $t and idx = $idx, using mean policy instead"))
    end

    res = Optim.optimize(diffobjective, constraints, u₀, IPNewton(), optoptions)

    if Optim.converged(res)
        u .= Optim.minimizer(res)
    else # If it does not converge we take the mean of the policy in the neighbourhood
        if verbose ≥ 1 
            @warn "Constrained optimisation has not converged at t = $t and idx = $idx"
        end

        unit = oneunit(idx)
        L = max(minimum(CartesianIndices(G)), idx - unit)
        R = min(maximum(CartesianIndices(G)), idx + unit)

        u .= mean(policy[L:R, :])
    end
end

function backwardstep!(Δts, F::NTuple{2, Matrix{Float64}}, policy, cluster, model::AbstractModel, G; αfactor = 1.5, allownegative = false, optargs...)
    Fₜ, Fₜ₊ₕ = F

    @threads for (i, δt) in cluster
        indices = CartesianIndices(G)

        idx = indices[i]
        t = model.economy.τ - δt
        M = exp(G.X[idx].m)
        ᾱ = γ(t, model.calibration) + δₘ(M, model.hogg)
        u₀ = copy(policy[idx, :])

        objective = @closure u -> begin
            Fᵉₜ, Δt = markovstep(t, idx, Fₜ₊ₕ, u[2], model, G)
            return logcost(Fᵉₜ, t, G.X[idx], Δt, u, model)
        end

        diffobjective = TwiceDifferentiable(objective, u₀; autodiff = :forward)

        # Solve first unconstrained problem
        openinterval = TwiceDifferentiableConstraints([0., 0.], [1., Inf])
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
        Δts[i] = last(markovstep(t, idx, Fₜ₊ₕ, policy[idx, 2], model, G))
    end
end

function backwardsimulation!(F::NTuple{2, Matrix{Float64}}, policy, model::AbstractModel, G; kwargs...)
    queue = DiagonalRedBlackQueue(G)
    backwardsimulation!(queue, F, policy, model, G; kwargs...)
end

function backwardsimulation!(queue::PartialQueue, F::NTuple{2, Matrix{Float64}}, policy, model::AbstractModel, G; verbose = 0, cachepath = nothing, cachestep = 0.25, overwrite = false, tstop = 0., tcache = model.economy.τ, stepkwargs...)
    
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
        
        clusters = dequeue!(queue)
        for cluster in clusters
            backwardstep!(Δts, F, policy, cluster, model, G; verbose, stepkwargs...)

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

function computebackward(model::AbstractModel, G; outdir = "data", kwargs...)
    terminalresults = loadterminal(model; outdir)
    computebackward(terminalresults, model, G; outdir, kwargs...)
end
function computebackward(terminalresults, model::AbstractModel, G; verbose = 0, withsave = true, outdir = "data", iterkwargs...)
    F̄, terminalconsumption, terminalG = terminalresults
    Fₜ₊ₕ = interpolateovergrid(terminalG, G, F̄);
    Fₜ = similar(Fₜ₊ₕ)
    F = (Fₜ, Fₜ₊ₕ)

    policy = Array{Float64}(undef, size(G)..., 2)
    policy[:, :, 1] .= interpolateovergrid(terminalG, G, terminalconsumption)
    policy[:, :, 2] .= γ(model.economy.τ, model.calibration)

    if withsave
        folder = SIMPATHS[typeof(model)]
        cachefolder = joinpath(outdir, folder)
        if !isdir(cachefolder) mkpath(cachefolder) end
        
        filename = makefilename(model)
    end

    cachepath = withsave ? joinpath(cachefolder, filename) : nothing
    backwardsimulation!(F, policy, model, G; verbose, cachepath, iterkwargs...)

    return F, policy
end
