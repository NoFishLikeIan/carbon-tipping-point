function terminalcost(τ, F′, Δt, Xᵢ::Point, χ, model::M) where {T,D<:GrowthDamages{T},P<:Preferences{T},M<:AbstractModel{T,D,P}}
    δ = terminaloutputfct(τ, Δt, Xᵢ, χ, model)
    return g(χ, δ * F′, Δt, model.preferences)
end
function terminalcost(τ, F′, Δt, Xᵢ::Point, χ, model::M) where {T,D<:LevelDamages{T},P<:Preferences{T},M<:AbstractModel{T,D,P}}
    δ = terminaloutputfct(τ, Δt, Xᵢ, χ, model)
    damage = d(Xᵢ.T, Xᵢ.m, model)
    return g(damage * χ, δ * F′, Δt, model.preferences)
end

function logterminalcost(τ, F′, Δt, Xᵢ::Point, χ, model::M) where {T,D<:GrowthDamages{T},P<:Preferences{T},M<:AbstractModel{T,D,P}}
    δ = logterminaloutputfct(τ, Δt, Xᵢ, χ, model)
    logF′ = δ + F′
    return logg(χ, logF′, Δt, model.preferences)
end

function terminaltimestep(idx, Δtmax, model::M, G) where M<:AbstractModel
    ΔT, Δm = G.Δ
    σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * ΔT))^2
    σₘ² = (model.hogg.σₘ / Δm)^2

    Xᵢ = G.X[idx]
    dT = μ(Xᵢ.T, Xᵢ.m, model) / (model.hogg.ϵ * ΔT)
    Q = σₘ² + σₜ² + G.h * abs(dT)

    return min(G.h^2 / Q, Δtmax)
end
function terminaldriftstep(idx, F̄, Δtmax, model::M, G) where M<:AbstractModel
    L, R = extrema(G)
    ΔT, Δm = G.Δ

    σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * ΔT))^2
    σₘ² = (model.hogg.σₘ / Δm)^2

    Xᵢ = G.X[idx]
    dT = μ(Xᵢ.T, Xᵢ.m, model) / (model.hogg.ϵ * ΔT)

    # Neighbouring nodes
    FᵢT₊, FᵢT₋, Fᵢm₊, Fᵢm₋ = getneighours(F̄, idx, L, R)

    dTF = G.h * abs(dT) * ifelse(dT > 0, FᵢT₊, FᵢT₋) + σₜ² * (FᵢT₊ + FᵢT₋) / 2
    dmF = σₘ² * (Fᵢm₊ + Fᵢm₋) / 2

    Q = σₘ² + σₜ² + G.h * abs(dT)
    F′ = (dTF + dmF) / Q
    Δt = min(G.h^2 / Q, Δtmax) # Ensure the time step is not too large

    return F′, Δt
end
function terminalmarkovstep(idx, F̄, Δtmax, model::M, G) where M<:Union{TippingModel,LinearModel}
    terminaldriftstep(idx, F̄, Δtmax, model, G)
end
function terminalmarkovstep(idx, F̄, Δtmax, model::JumpModel, G)
    Fᵈ, Δt = terminaldriftstep(idx, F̄, Δtmax, model, G)

    # Update with jump
    R = maximum(CartesianIndices(F̄))
    Xᵢ = G.X[idx]
    πᵢ = intensity(Xᵢ.T, model.hogg, model.jump)
    qᵢ = increase(Xᵢ.T, model.hogg, model.jump)

    steps = floor(Int, div(qᵢ, G.Δ.T * G.h))
    weight = qᵢ / (G.Δ.T * G.h)

    Fʲ = F̄[min(idx + steps * Idx[1], R)] * (1 - weight) +
         F̄[min(idx + (steps + 1) * Idx[1], R)] * weight

    F′ = Fᵈ + πᵢ * Δt * (Fʲ - Fᵈ)

    return F′, Δt
end

function terminaljacobi!(state::DPState, model, G; iterkwargs...)
    terminaljacobi!(state.valuefunction, state.policystate, state.timestate, model, G; iterkwargs...)
end
function terminaljacobi!(valuefunction::ValueFunction{T}, policystate::PolicyState, timestate::Time, model, G; indices=CartesianIndices(G), ω=1., opttol=1e-5, Δtmax=1 / 100) where T
    @inbounds @threads for idx in indices
        F′, Δt = terminalmarkovstep(idx, valuefunction.Fₜ, Δtmax, model, G)
        Xᵢ = G.X[idx]

        # Optimal control
        objective = χ -> logterminalcost(timestate.τ, F′, Δt, Xᵢ, χ, model)
        Fᵢ, χ = gssmin(objective, zero(T), one(T); tol=opttol)
        Fᵢ′ = ω * Fᵢ + (1 - ω) * valuefunction.Fₜ[idx]

        valuefunction.error[idx] = abs(Fᵢ′ - valuefunction.Fₜ[idx])
        valuefunction.Fₜ[idx] = Fᵢ′
        policystate.policy[idx].χ = χ
    end
end

function vfi(model::M, calibration::Calibration, G; kwargs...) where M<:AbstractModel
    state = DPState(calibration, G)
    return vfi!(state, model, G; kwargs...)
end
function vfi!(state::DPState, model::M, G; tol=1e-3, maxiter=10_000, verbose=0, indices=CartesianIndices(G), alternate=false, optkwargs...) where M<:AbstractModel
    (verbose ≥ 1) && println("Starting iterations...")
    magnitude = floor(Int, -log10(tol))

    indicescollection = alternate ?
                        (indices, reverse(indices, dims=1), reverse(indices, dims=2), reverse(indices)) : (indices,)

    alternatem = length(indicescollection)

    for iter in 1:maxiter
        iterindices = indicescollection[1+iter%alternatem]
        terminaljacobi!(state, model, G; indices=iterindices, optkwargs...)

        maxerror = maximum(state.valuefunction.error)

        if isnan(maxerror)
            @warn "NaN error detected. Exiting."
            state.valuefunction.Fₜ₊ₕ .= state.valuefunction.Fₜ
            return state
        end

        if maxerror < tol
            (verbose ≥ 1) && printjacobiterminal(maxerror, iter, magnitude)
            state.valuefunction.Fₜ₊ₕ .= state.valuefunction.Fₜ
            return state
        end

        if (verbose ≥ 2) && (!alternate || iter % alternatem == 0)
            printjacobi(maxerror, iter, maxiter, magnitude)
        end
    end

    (verbose ≥ 1) && @warn @sprintf "Convergence failed, did not reach %f tolerance in %i iterations." tol maxiter

    state.valuefunction.Fₜ₊ₕ .= state.valuefunction.Fₜ
    return state
end

function computeterminal(model::M, calibration::Calibration, G::RegularGrid; verbose=0, withsave=true, outdir="data", overwrite=false, addpath="", iterkwargs...) where M<:AbstractModel

    if withsave
        folder = simpaths(model)
        savefolder = joinpath(outdir, folder, "terminal", addpath)
        if !isdir(savefolder)
            mkpath(savefolder)
        end

        filename = makefilename(model)
        savepath = joinpath(savefolder, filename)

        if isfile(savepath)
            if overwrite
                (verbose ≥ 1) && @warn "Removing file $savepath.\n"

                rm(savepath)
            else
                (verbose ≥ 1) && @warn "File $savepath already exists. If you want to overwrite it pass overwrite = true. Will copy the results into `F` and `policy`.\n"

                state, G = loadterminal(model; outdir, addpath)

                return state, G
            end
        end
    end

    state = vfi(model, calibration, G; verbose=verbose, iterkwargs...)

    if withsave
        (verbose ≥ 1) && println("Saving solution into $savepath...")
        jldsave(savepath; state, G)
    end

    return state, G
end