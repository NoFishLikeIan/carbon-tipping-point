using Distributed: @everywhere, @distributed, @sync
using SharedArrays: SharedArray, SharedMatrix, SharedVector
using DataStructures: PriorityQueue, dequeue!, enqueue!, peek

using Model, Grid

@everywhere begin
    using Model, Grid
    using JLD2, DotEnv
    using UnPack: @unpack
    using ZigZagBoomerang: dequeue!
    using Base: Order
    using FastClosures: @closure
    using Optim
end

@everywhere include("chain.jl")

@everywhere function updateᾱ!(constraints::TwiceDifferentiableConstraints, ᾱ)
    constraints.bounds.bx[4] = ᾱ
end

function backwardstep!(Δts, F, policy, cluster, model::AbstractModel, G)
    indices = CartesianIndices(G)
    constraints = TwiceDifferentiableConstraints([0., 0.], [1., 1.])

    @sync @distributed for (i, δt) in cluster
        idx = indices[i]
        Xᵢ = G.X[idx]

        t = model.economy.τ - δt
        ᾱ = γ(t, model.calibration) + δₘ(exp(Xᵢ.m), model.hogg)
        updateᾱ!(constraints, ᾱ)

        objectivefn = @closure u -> begin
            F′, Δt = markovstep(t, idx, F, u, model, G)
            c = cost(F′, t, Xᵢ, Δt, u, model)
            return c
        end

        u₀ = clamp.(policy[idx, :], 
            constraints.bounds.bx[[1, 3]] .+ 1e-5, 
            constraints.bounds.bx[[2, 4]] .- 1e-5) # Prevents u₀ to close to constraint

        diffobj = TwiceDifferentiable(objectivefn, u₀; autodiff = :forward)
        res = Optim.optimize(diffobj, constraints, u₀, IPNewton())
        
        !Optim.converged(res) && @warn "Optim has not converged at t = $t and idx = $idx"

        policy[idx, :] .= Optim.minimizer(res)
        F[idx] = Optim.minimum(res)
        Δts[i] = last(markovstep(t, idx, F, policy[idx, :], model, G))
    end
end

"Backward simulates from F̄ down to F₀, using the albedo model. It assumes that the passed F ≡ F̄"
function backwardsimulation!(F, policy, model::AbstractModel, G; verbose = false, cachepath = nothing, cachestep = 0.25, overwrite = false, tstop = 0., tcache = last(model.calibration.tspan), stepkwargs...)
    verbose && println("Starting backward simulation...")
     
    savecache = !isnothing(cachepath)
    if savecache
        if isfile(cachepath) 
            if overwrite 
                verbose && @warn "Removing file $cachepath.\n"

                rm(cachepath)
            else 
                verbose && @warn "File $cachepath already exists. If you want to overwrite it pass overwrite = true. Will copy the results into `F` and `policy`.\n"

                _, Fcache, policycache = loadtotal(model; datapath)

                F .= Fcache[:, :, 1]
                policy .= policycache[:, :, 1]

                return F, policy
            end
        end

        cachefile = jldopen(cachepath, "w+")
        cachefile["G"] = G
    end

    queue = DiagonalRedBlackQueue(G)
    Δts = SharedVector(zeros(N^2))
    passcounter = 1

    while !isqempty(queue)
        tmin = model.economy.τ - minimum(queue.vals)
        if verbose
            @printf("Pass %i, cluster minimum time = %.4f...\r", passcounter, tmin)
        end

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
            verbose && println("\nSaving cache at $tcache...")
            group = JLD2.Group(cachefile, "$tcache")
            group["F"] = F
            group["policy"] = policy
            tcache = tcache - cachestep 
        end
    end

    if savecache close(cachefile) end

    return F, policy
end

function computebackward(model::AbstractModel, G; datapath = "data", kwargs...)
    terminalresults = loadterminal(model; datapath)
    computebackward(terminalresults, model, G; datapath, kwargs...)
end
function computebackward(terminalresults, model::AbstractModel, G; verbose = false, withsave = true, datapath = "data", iterkwargs...)
    F̄, terminalconsumption, terminalG = terminalresults
    F = SharedMatrix(interpolateovergrid(terminalG, G, F̄));

    policy = SharedArray{Float64}(size(G)..., 2)
    policy[:, :, 1] .= interpolateovergrid(terminalG, G, terminalconsumption)
    policy[:, :, 2] .= γ(model.economy.τ, model.calibration)

    if withsave
        folder = SIMPATHS[typeof(model)]
        cachefolder = joinpath(datapath, folder)
        if !isdir(cachefolder) mkpath(cachefolder) end
        
        filename = makefilename(model)
    end

    cachepath = ifelse(withsave, joinpath(cachefolder, filename), nothing)
    backwardsimulation!(F, policy, model, G; verbose, cachepath, iterkwargs...)

    return F, policy
end
