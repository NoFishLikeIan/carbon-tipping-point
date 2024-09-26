using JLD2
using Printf: @printf

using Model, Grid
using FastClosures: @closure
using Printf: @printf
using ZigZagBoomerang: dequeue!
using Base: Threads
using SciMLBase: successful_retcode
using Optimization
using OptimizationNLopt, OptimizationMultistartOptimization
using OptimizationOptimJL

include("chain.jl")


function backwardstep!(Δts, F, policy, cluster, model::AbstractModel, G; allownegative = false, ad = Optimization.AutoForwardDiff(), solver = LBFGS(), tiktak = MultistartOptimization.TikTak(100))
    objective = @closure (u, p) -> begin
        t, idx = p

        F′, Δt = markovstep(t, idx, F, u[2], model, G)
        return logcost(F′, t, G.X[idx], Δt, u, model)
    end
    fn = OptimizationFunction(objective, ad)

    Threads.@threads for (i, δt) in cluster
        indices = CartesianIndices(G)

        idx = indices[i]
        t = model.economy.τ - δt
        u₀ = policy[idx, :]

        ub = if allownegative
            [1., 100.]
        else
            ᾱ = γ(t, model.calibration) + δₘ(exp(G.X[idx].m), model.hogg)
            [1., ᾱ]
        end

        prob = OptimizationProblem(fn, u₀, (t, idx), lb = [0., 0.], ub = ub)
        sol = solve(prob, tiktak, solver)

        hasconverged = successful_retcode(sol.retcode)
        
        !hasconverged && @warn "Optimisation has not converged at t = $t and idx = $idx"

        policy[idx, :] .= sol.u
        F[idx] = exp(sol.objective)
        Δts[i] = last(markovstep(t, idx, F, sol.u[2], model, G))
    end
end

"Backward simulates from F̄ down to F₀, using the albedo model. It assumes that the passed F ≡ F̄"
function backwardsimulation!(F, policy, model::AbstractModel, G; verbose = 0, cachepath = nothing, cachestep = 0.25, overwrite = false, tstop = 0., tcache = 150., stepkwargs...)     
    savecache = !isnothing(cachepath)
    if savecache
        if isfile(cachepath) 
            if overwrite 
                (verbose ≥ 1) && @warn "Removing file $cachepath.\n"
                rm(cachepath)

                cachefile = jldopen(cachepath, "w+")
                cachefile["G"] = G
                cachefile["model"] = model
            else 
                (verbose ≥ 1) && @warn "File $cachepath already exists. If you want to overwrite it pass overwrite = true. Will copy the results into `F` and `policy`.\n"

                _, Fcache, policycache, Gcache, _ = loadtotal(cachepath)

                F .= interpolateovergrid(Gcache, G, Fcache[:, :, 1])

                policy[:, :, 1] = interpolateovergrid(Gcache, G, policycache[:, :, 1, 1])
                policy[:, :, 2] = interpolateovergrid(Gcache, G, policycache[:, :, 2, 1])

                return F, policy
            end
        else
            cachefile = jldopen(cachepath, "w+")
            cachefile["G"] = G
            cachefile["model"] = model
        end
    end

    queue = DiagonalRedBlackQueue(G)
    Δts = zeros(N^2)
    passcounter = 1

    while !isempty(queue)
        tmin = model.economy.τ - minimum(queue.vals)

        if (verbose ≥ 2) || ((verbose ≥ 1) && (passcounter % 500 == 0))
            @printf("%s: pass %i, cluster minimum time = %.4f\n", now(), passcounter, tmin)
            flush(stdout)
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
            if (verbose ≥ 2)
                println("-- Saving cache at $tcache")
            end

            group = JLD2.Group(cachefile, "$tcache")
            group["F"] = F
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
