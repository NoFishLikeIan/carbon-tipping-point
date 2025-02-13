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

using Dates: now

include("chain.jl")

const defaultoptoptions = Optim.Options(g_tol = 1e-12, iterations = 100_000)

"""
Optimise `diffobjective` and stores the minimiser in u. If the optimisation does not converge, it takes the mean of the policy in the neighbourhood.
"""
function fallbackoptimisation!(u,
    diffobjective, u₀, 
    idx, t, policy, 
    constraints::TwiceDifferentiableConstraints, G::RegularGrid)

    if !Optim.isinterior(constraints, u₀)
        @warn "Initial guess is not in the interior of the constraints at t = $t and idx = $idx, using mean policy instead"
        
        unit = oneunit(idx)
        L = max(minimum(CartesianIndices(G)), idx - unit)
        R = min(maximum(CartesianIndices(G)), idx + unit)

        u .= mean(@view policy[L:R, :])
    else
        res = Optim.optimize(diffobjective, constraints, u₀, IPNewton(), defaultoptoptions)

        if Optim.converged(res)
            u .= Optim.minimizer(res)
        else # If it does not converge we take the mean of the policy in the neighbourhood 
            @warn "Constrained optimisation has not converged at t = $t and idx = $idx"

            unit = oneunit(idx)
            L = max(minimum(CartesianIndices(G)), idx - unit)
            R = min(maximum(CartesianIndices(G)), idx + unit)

            u .= mean(@view policy[L:R, :])
        end
    end
end

function backwardstep!(Δts, F::NTuple{2, Matrix{Float64}}, policy, cluster, model::AbstractModel, calibration::Calibration, G)
    Fₜ, Fₜ₊ₕ = F

    @threads for (i, δt) in cluster
        @inbounds begin
        indices = CartesianIndices(G)

        idx = indices[i]
        t = model.economy.τ - δt
        M = exp(G.X[idx].m)

        objective = @closure u -> begin
            Fᵉₜ, Δt = markovstep(t, idx, Fₜ₊ₕ, u[2], model, calibration, G)
            return logcost(Fᵉₜ, t, G.X[idx], Δt, u, model, calibration)
        end

        ᾱ = γ(t, calibration) + δₘ(M, model.hogg)
        u₀ = [0.5, ᾱ / 2.]
        diffobjective = TwiceDifferentiable(objective, u₀; autodiff = :forward)
        rectangle = TwiceDifferentiableConstraints([0., 0.], [1., ᾱ])
        
        optimum = similar(u₀)
        fallbackoptimisation!(optimum, diffobjective, u₀, idx, t, policy, rectangle, G)
        
        objectiveminimum = objective(optimum)

        policy[idx, :] .= optimum
        Fₜ[idx] = exp(objectiveminimum)
        Δts[i] = markovstep(t, idx, Fₜ₊ₕ, optimum[2], model, calibration, G)[2]
        end
    end 
end

function backwardsimulation!(F::NTuple{2, Matrix{Float64}}, policy, model::AbstractModel, calibration::Calibration, G; kwargs...)
    queue = DiagonalRedBlackQueue(G)
    backwardsimulation!(queue, F, policy, model, calibration, G; kwargs...)
end

function backwardsimulation!(queue::PartialQueue, F::NTuple{2, Matrix{Float64}}, policy, model::AbstractModel, calibration::Calibration, G; verbose = 0, cachepath = nothing, cachestep = 0.25, overwrite = false, tstop = 0., tcache = model.economy.τ)
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
        
        clusters = dequeue!(queue)
        for cluster in clusters
            backwardstep!(Δts, F, policy, cluster, model, calibration, G)

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
function computebackward(terminalresults, model::AbstractModel, calibration::Calibration, G; verbose = 0, withsave = true, outdir = "data", iterkwargs...)
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

    backwardsimulation!(F, policy, model, calibration, G; verbose, cachepath, iterkwargs...)

    return F, policy
end
