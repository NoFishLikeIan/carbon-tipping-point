using Model, Grid
using JLD2
using UnPack: @unpack
using ZigZagBoomerang: dequeue!
using DataStructures: PriorityQueue, dequeue!, enqueue!, peek
using Base: Order
using FastClosures: @closure
using Optim
using Printf: @printf

include("chain.jl")

const defaultoptim = Optim.Options(
    g_tol = 1e-12, 
    allow_f_increases = true, 
    iterations = 100_000);

function backwardstep!(Δts, F, policy, cluster, model::AbstractModel, G; allownegative = false, options = defaultoptim)
    for (i, δt) in cluster
        indices = CartesianIndices(G)
        idx = indices[i]
        Xᵢ = G.X[idx]

        t = model.economy.τ - δt
        u₀ = policy[idx, :]

        if allownegative
            constraints = TwiceDifferentiableConstraints([0., 0.], [1., 1.])
            u₀ = Optim.isinterior(constraints, u₀) ? u₀ : [0.5, 0.5]
        else
            ᾱ = γ(t, model.calibration) + δₘ(exp(Xᵢ.m), model.hogg)
            constraints = TwiceDifferentiableConstraints([0., 0.], [1., ᾱ])
            u₀ = Optim.isinterior(constraints, u₀) ? u₀ : [0.5, ᾱ / 2]
        end

        objective = @closure u -> begin
            F′, Δt = markovstep(t, idx, F, u, model, G)
            logcost(F′, t, Xᵢ, Δt, u, model)
        end

        diffobj = TwiceDifferentiable(objective, u₀; autodiff = :forward)
        res = Optim.optimize(diffobj, constraints, u₀, IPNewton(), options)
        
        !Optim.converged(res) && @warn "Optim has not converged at t = $t and idx = $idx"

        u = Optim.minimizer(res)
        u[2] = ifelse(abs(u[2]) < 1e-10, 0., u[2]) # Round smallest numbers down

        policy[idx, :] .= u
        F[idx] = exp(Optim.minimum(res))
        Δts[i] = last(markovstep(t, idx, F, u, model, G))
    end
end

"Backward simulates from F̄ down to F₀, using the albedo model. It assumes that the passed F ≡ F̄"
function backwardsimulation!(F, policy, model::AbstractModel, G; verbose = 0, cachepath = nothing, cachestep = 0.25, overwrite = false, tstop = 0., tcache = last(model.calibration.tspan), stepkwargs...)     
    savecache = !isnothing(cachepath)
    if savecache
        if isfile(cachepath) 
            if overwrite 
                (verbose ≥ 1) && @warn "Removing file $cachepath.\n"
                rm(cachepath)

                cachefile = jldopen(cachepath, "w+")
                cachefile["G"] = G
            else 
                (verbose ≥ 1) && @warn "File $cachepath already exists. If you want to overwrite it pass overwrite = true. Will copy the results into `F` and `policy`.\n"

                _, Fcache, policycache, Gcache = loadtotal(cachepath)

                F .= interpolateovergrid(Gcache, G, Fcache[:, :, 1])

                policy[:, :, 1] = interpolateovergrid(Gcache, G, policycache[:, :, 1, 1])
                policy[:, :, 2] = interpolateovergrid(Gcache, G, policycache[:, :, 2, 1])

                return F, policy
            end
        else
            cachefile = jldopen(cachepath, "w+")
            cachefile["G"] = G
        end
    end

    queue = DiagonalRedBlackQueue(G)
    Δts = SharedVector(zeros(N^2))
    passcounter = 1

    while !isqempty(queue)
        tmin = model.economy.τ - minimum(queue.vals)
        (verbose ≥ 2) && @printf("Pass %i, cluster minimum time = %.4f...\r", passcounter, tmin)

        clusters = dequeue!(queue)

        for cluster in clusters
            backwardstep!(Δts, F, policy, cluster, model, G; stepkwargs...)

            for i in first.(cluster)
                if queue[i] ≤ model.economy.τ - tstop
                    queue[i] += Δts[i]
                end
            end
        end

        passcounter += 1
        
        if savecache && tmin ≤ tcache
            (verbose ≥ 2) && println("\nSaving cache at $tcache...")
            group = JLD2.Group(cachefile, "$tcache")
            group["F"] = F
            group["policy"] = policy
            tcache = tcache - cachestep 
        end
    end

    if savecache 
        close(cachefile) 
        (verbose ≥ 1) && println("\nSaved cached file into $cachepath")
    end

    return F, policy
end

function computebackward(model::AbstractModel, G; outdir = "data", kwargs...)
    terminalresults = loadterminal(model; outdir)
    computebackward(terminalresults, model, G; outdir, kwargs...)
end
function computebackward(terminalresults, model::AbstractModel, G; verbose = 0, withsave = true, outdir = "data", iterkwargs...)
    F̄, terminalconsumption, terminalG = terminalresults
    F = interpolateovergrid(terminalG, G, F̄);

    policy = Array{Float64}(undef, size(G)..., 2)
    policy[:, :, 1] .= interpolateovergrid(terminalG, G, terminalconsumption)
    policy[:, :, 2] .= γ(model.economy.τ, model.calibration)

    if withsave
        folder = SIMPATHS[typeof(model)]
        cachefolder = joinpath(outdir, folder)
        if !isdir(cachefolder) mkpath(cachefolder) end
        
        filename = makefilename(model)
    end

    cachepath = ifelse(withsave, joinpath(cachefolder, filename), nothing)
    backwardsimulation!(F, policy, model, G; verbose, cachepath, iterkwargs...)

    return F, policy
end
