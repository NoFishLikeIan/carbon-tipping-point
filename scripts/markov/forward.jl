function backwardstep!(Δts, F::Matrix{Float64}, χitp::Extrapolation, αitp::Extrapolation, cluster, model::AbstractModel, calibration::Calibration, G)
    @threads for (i, δt) in cluster
        indices = CartesianIndices(G)

        idx = indices[i]
        t = model.economy.τ - δt
        Xᵢ = G.X[idx]
        
        u = (χitp(Xᵢ.T, Xᵢ.m, t), αitp(Xᵢ.T, Xᵢ.m, t))

        F′, Δt = markovstep(t, idx, F, u[2], model, calibration, G)
        F[idx] = cost(F′, t, Xᵢ, Δt, u, model, calibration)
        Δts[i] = Δt
    end
end

function backwardsimulation!(queue::PartialQueue, F::Matrix{Float64}, χitp::Extrapolation, αitp::Extrapolation, model::AbstractModel, calibration::Calibration, G; verbose = 0, tstop = 0.)
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
            backwardstep!(Δts, F, χitp, αitp, cluster, model, calibration, G)

            indices = first.(cluster)

            for i in indices
                if queue[i] ≤ model.economy.τ - tstop
                    queue[i] += Δts[i]
                end
            end
        end
    end
end

function backwardsimulation!(F::Matrix{Float64}, χitp::Extrapolation, αitp::Extrapolation, model::AbstractModel, calibration::Calibration, G; kwargs...)
    queue = DiagonalRedBlackQueue(G)
    backwardsimulation!(queue, F, χitp, αitp, model, calibration, G; kwargs...)
end

function computebackward(χitp::Extrapolation, αitp::Extrapolation, model::AbstractModel, calibration::Calibration; outdir = "", kwargs...)
    terminalresults = loadterminal(model; outdir)
    computebackward(terminalresults, χitp, αitp, model, calibration; kwargs...)
end

function computebackward(terminalresults, χitp::Extrapolation, αitp::Extrapolation, model::AbstractModel, calibration::Calibration; verbose = 0, iterkwargs...)
    F̄, _, G = terminalresults
    F = copy(F̄)

    backwardsimulation!(F, χitp, αitp, model, calibration, G; verbose, iterkwargs...)

    return F
end
