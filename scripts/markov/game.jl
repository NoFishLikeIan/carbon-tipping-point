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
    idx, k, i, t, policies, 
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

        u[1] = mean(policies[1, L:R, k, i])
        u[2] = mean(policies[2, L:R, k, i])
    end
end

function backwardstep!(Δts, F, policies, cluster, model::AbstractGameModel, G; αfactor = 1.5, allownegative = false, optargs...)
    Fₜ, Fₜ₊ₕ = F
    indices = CartesianIndices(G)
    models = breakgamemodel(model)

    τ = models[1].economy.τ
    M = size(Fₜ, 3)
    unit = range(0, 1; length = M)

    for (l, δt) in cluster
        idx = indices[l]
        Xᵢ = G.X[idx]

        t = τ - δt
        δₘₜ = δₘ(exp(Xᵢ.m), model.hogg)
        ᾱ = γ(t, model.regionalcalibration) .+ δₘₜ

        Δtₗ = Inf

        for (i, regionalmodel) in enumerate(models)
            j = ifelse(i == 1, 2, 1)
            A = ᾱ[j] * unit # Action space of opponent

            for (k, αⱼ) in enumerate(A)
                Fᵏₜ₊ₕ = @view Fₜ₊ₕ[:, :, k, i]
                u₀ = copy(policies[:, idx, k, i])

                objective = @closure u -> begin
                    Fᵉₜ, Δt = markovstep(t, idx, Fᵏₜ₊ₕ, u[2], αⱼ, regionalmodel, G)
                    return logcost(Fᵉₜ, t, G.X[idx], Δt, u, regionalmodel)
                end

                diffobjective = TwiceDifferentiable(objective, u₀; autodiff = :forward)

                # Solve first unconstrained problem
                openinterval = TwiceDifferentiableConstraints([0., 0.], [1., Inf])
                u₀[2] = αfactor * ᾱ[i]
                optimum = similar(u₀)

                fallbackoptimisation!(optimum, diffobjective, u₀, idx, k, i, t, policies, openinterval, G; optargs...)

                if !allownegative # Solve constrained problem with χ from unconstrained problem
                    u₀[1] = optimum[1]
                    u₀[2] = min(optimum[2], ᾱ[i] / 2) # Guarantees u₀ ∈ U
                    constraints = TwiceDifferentiableConstraints([0., 0.], [1., ᾱ[i]])
        
                    fallbackoptimisation!(optimum, diffobjective, u₀, idx, k, i, t, policies, constraints, G; optargs...)
                end
        
                objectiveminimum = objective(optimum)

                policies[:, idx, k, i] .= optimum
                Fₜ[idx, k, i] = exp(objectiveminimum)
                timestep = last(markovstep(t, idx, Fᵏₜ₊ₕ, optimum[2], αⱼ, regionalmodel, G))

                Δtₗ = min(timestep, Δtₗ)
            end
        end

        Δts[l] = Δtₗ
    end
end

function backwardsimulation!(F::NTuple{2, Array{Float64, 4}}, policy, model::AbstractModel, G; kwargs...)
    queue = DiagonalRedBlackQueue(G)
    backwardsimulation!(queue, F, policy, model, G; kwargs...)
end

function backwardsimulation!(queue::PartialQueue, F::NTuple{2, Array{Float64, 4}}, policies, model::AbstractGameModel, G; verbose = false, cachepath = nothing, cachestep = 0.25, overwrite = false, tstop = 0., tcache = last(model.regionalcalibration.calibration.tspan), allownegative = false, stepkwargs...)
    verbose && println("Starting backward simulation...")
     
    savecache = !isnothing(cachepath)
    if savecache
        if isfile(cachepath)
            if overwrite 
                verbose && @warn "Removing file $cachepath.\n"

                rm(cachepath)
            else 
                verbose && @warn "File $cachepath already exists. If you want to overwrite it pass overwrite = true. Will copy the results into `F` and `policy`.\n"

                _, Fcache, policycache, _, _ = loadtotal(model; allownegative)

                F .= Fcache[:, :, 1]
                policy .= policycache[:, :, 1]

                return F, policy
            end
        end

        cachefile = jldopen(cachepath, "w+")
        cachefile["G"] = G
    end

    Δts = Vector{Float64}(undef, prod(size(G)))
    τ = first(model.economy).τ

    passcounter = 1

    while !isempty(queue)
        tmin = τ - minimum(queue.vals)
        clusters = dequeue!(queue)

        for cluster in clusters
            backwardstep!(Δts, F, policies, cluster, model, G; stepkwargs...)

            verbose && print("Cluster minimum time = $tmin...\r")

            for i in first.(cluster)
                if queue[i] ≤ τ - tstop
                    queue[i] += Δts[i]
                end
            end
        end
        
        if savecache && tmin ≤ tcache
            verbose && println("\nSaving cache at $tcache...")
            group = JLD2.Group(cachefile, "$tcache")
            group["F"] = F
            group["policies"] = policies
            tcache = tcache - cachestep 
        end
    end

    if savecache close(cachefile) end

    return F, policies
end

function computebackward(model::AbstractGameModel, G; datapath = "data", addpaths = ["high", "low"], kwargs...)
    highmodel, lowmodel = breakgamemodel(model)
    F̄high, χhigh, terminalG = loadterminal(highmodel; outdir = datapath, addpath = addpaths[1])
    F̄low, χlow, _ = loadterminal(lowmodel; outdir = datapath, addpath = addpaths[2])

    F̄ = cat(F̄high, F̄low; dims = 3)
    terminalpolicy = cat(χhigh, χlow; dims = 3)
    
    terminalresults = F̄, terminalpolicy, terminalG
    
    computebackward(terminalresults, model, G; datapath, kwargs...)
end

TerminalResults = Tuple{Array{Float64, 3}, Array{Policy, 3}, RegularGrid}

function computebackward(terminalresults::TerminalResults, model::AbstractGameModel, G; verbose = false, withsave = true, datapath = "data", allownegative = false, M = 10, iterkwargs...)
    F̄, terminalconsumption, terminalG = terminalresults
    
    N₁, N₂ = size(G)
    nplayers = size(F̄, 3)

    Fₜ₊ₕ = Array{Float64}(undef, N₁, N₂, M, nplayers)
    policies = Array{Float64}(undef, 2, N₁, N₂, M, nplayers)

    for k in 1:nplayers
        Fₜ₊ₕ[:, :, :, k] .= interpolateovergrid(terminalG, G, F̄[:, :, k])
        policies[1, :, :, :, :, :] .= interpolateovergrid(terminalG, G, terminalconsumption[:, :, k])
        policies[2, :, :, :, :, :] .= γ(model.economy[k].τ, model.regionalcalibration)[k]
    end

    Fₜ = similar(Fₜ₊ₕ)
    F = (Fₜ, Fₜ₊ₕ);

    if withsave
        folder = SIMPATHS[typeof(model)]
        controltype = ifelse(allownegative, "allownegative", "nonnegative")
        cachefolder = joinpath(datapath, folder, controltype, "cache")
        if !isdir(cachefolder) mkpath(cachefolder) end
        
        filename = makefilename(model)
    end

    cachepath = withsave ? joinpath(cachefolder, filename) : nothing
    
    backwardsimulation!(F, policies, model, G; verbose, cachepath, allownegative, iterkwargs...)

    return F, policies
end