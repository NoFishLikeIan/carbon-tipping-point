using JLD2
using Printf: @printf

using Model, Grid
using FastClosures: @closure
using Printf: @printf
using ZigZagBoomerang: dequeue!
using Base.Threads: @threads
using SciMLBase: successful_retcode
using Optimization
using OptimizationNLopt, OptimizationMultistartOptimization
using OptimizationOptimJL

include("chain.jl")


function backwardstep!(Δts, F::NTuple{2, Matrix{Float64}}, policy, cluster, model::AbstractModel, G; allownegative = false, ad = Optimization.AutoForwardDiff(), solver = LBFGS(), tiktak = MultistartOptimization.TikTak(100), αub = 100.)
    Fₜ, Fₜ₊ₕ = F

    objective = @closure (u, p) -> begin
        t, idx = p

        Fᵉₜ₊ₕ, Δt = markovstep(t, idx, Fₜ₊ₕ, u[2], model, G)
        return logcost(Fᵉₜ₊ₕ, t, G.X[idx], Δt, u, model)
    end

    fn = OptimizationFunction(objective, ad)

    @threads for (i, δt) in cluster
        ub = [1., αub]
        indices = CartesianIndices(G)

        idx = indices[i]
        t = model.economy.τ - δt
        u₀ = policy[idx, :]

        if !allownegative
            ᾱ = γ(t, model.calibration) + δₘ(exp(G.X[idx].m), model.hogg)
            ub[2] = ᾱ
        end

        prob = OptimizationProblem(fn, u₀, (t, idx), lb = zeros(2), ub = ub)
        sol = solve(prob, tiktak, solver)

        hasconverged = successful_retcode(sol.retcode)
        
        !hasconverged && @warn "Optimisation has not converged at t = $t and idx = $idx"

        policy[idx, :] .= sol.u
        Fₜ[idx] = exp(sol.objective)
        Δts[i] = last(markovstep(t, idx, Fₜ₊ₕ, sol.u[2], model, G))
    end
end

function backwardsimulation!(F::NTuple{2, Matrix{Float64}}, policy, model::AbstractModel, G; verbose = 0, cachepath = nothing, cachestep = 0.25, overwrite = false, tstop = 0., tcache = 150., stepkwargs...)
    
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

                F[1] .= interpolateovergrid(Gcache, G, Fcache[:, :, 1])

                policy[:, :, 1] = interpolateovergrid(Gcache, G, policycache[:, :, 1, 1])
                policy[:, :, 2] = interpolateovergrid(Gcache, G, policycache[:, :, 2, 1])

                return F[1], policy
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
                println("Saving cache at $tcache")
            end

            group = JLD2.Group(cachefile, "$tcache")
            group["F"] = F[1]
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

    return F[1], policy
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

    cachepath = ifelse(withsave, joinpath(cachefolder, filename), nothing)
    backwardsimulation!(F, policy, model, G; verbose, cachepath, iterkwargs...)

    return F, policy
end
