function terminalcost(τ, Fᵢ′, Xᵢ::Point, Δt, χ, model::AbstractModel{T, D, P}) where {T, D <: GrowthDamages{T}, P <: Preferences{T}}
    δ = terminaloutputfct(τ, Xᵢ, Δt, χ, model)
    return g(χ, δ * Fᵢ′, Δt, model.preferences)
end
function terminalcost(τ, Fᵢ′, Xᵢ::Point, Δt, χ, model::AbstractModel{T, D, P}) where {T, D <: LevelDamages{T}, P <: Preferences{T}}
    δ = terminaloutputfct(τ, Xᵢ, Δt, χ, model)
    damage = d(Xᵢ.T, Xᵢ.m, model)
    return g(damage * χ, δ * Fᵢ′, Δt, model.preferences)
end

function terminaltimestep(idx, model::AbstractModel, G)
    ΔT, Δm = G.Δ
    σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * ΔT))^2
    σₘ² = (model.hogg.σₘ / Δm)^2

    Xᵢ = G.X[idx]
    dT = μ(Xᵢ.T, Xᵢ.m, model) / (model.hogg.ϵ * ΔT)
    Q = σₘ² + σₜ² + G.h * abs(dT)

    return G.h^2 / Q
end
function terminaldriftstep(idx, F̄, model::AbstractModel, G)
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
    Δt = G.h^2 / Q

    return F′, Δt
end
function terminalmarkovstep(idx, F̄, model::M, G) where M <: Union{TippingModel, LinearModel} 
    terminaldriftstep(idx, F̄, model, G)
end
function terminalmarkovstep(idx, F̄, model::JumpModel, G)
    Fᵈ, Δt = terminaldriftstep(idx, F̄, model, G)

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
   terminaljacobi!(state.timestate.τ, state.valuefunction.Fₜ, state.policystate.policy, state.valuefunction.error, model, G; iterkwargs...)
end
function terminaljacobi!(τ, F::Matrix{T}, policy::Matrix{Policy{T}}, errors, model, G; indices = CartesianIndices(F)) where T
    @inbounds @threads for idx in indices
        F′, Δt = terminalmarkovstep(idx, F, model, G)
        Xᵢ = G.X[idx]

        # Optimal control
        objective = @closure χ -> terminalcost(τ, F′, Xᵢ, Δt, χ, model)
        Fᵢ, χ = gssmin(objective, zero(T), one(T); tol = eps(T))
        
        errors[idx] = abs(Fᵢ - F[idx]) / F[idx]
        F[idx] = Fᵢ
        policy[idx].χ = χ
    end
end

function vfi(model::AbstractModel, calibration, G; kwargs...)
    state = DPState(calibration, G)
    return vfi!(state, model, G; kwargs...)
end
function vfi!(state::DPState, model::AbstractModel, G; tol = 1e-3, maxiter = 10_000, verbose = 0, indices = CartesianIndices(G), alternate = false)
    (verbose ≥ 1) && println("Starting iterations...")
    magnitude = floor(Int, -log10(tol))

    for iter in 1:maxiter
        iterindices = (alternate && isodd(iter)) ? indices : reverse(indices)

        terminaljacobi!(state, model, G; indices = iterindices)

        maxerror = maximum(state.valuefunction.error)

        if isnan(maxerror)
            @warn "NaN error detected. Exiting."
            return state
        end

        if maxerror < tol
            (verbose ≥ 1) && printjacobiterminal(maxerror, iter, magnitude)
            return state
        end

        if (verbose ≥ 2) && (!alternate || isodd(iter))
            printjacobi(maxerror, iter, maxiter, magnitude)
        end
    end

    (verbose ≥ 1) && @warn @sprintf "Convergence failed, did not reach %f tolerance in %i iterations." tol maxiter
    return state
end

function computeterminal(model, G::RegularGrid; verbose = 0, withsave = true, outdir = "data", overwrite = false, addpath = "", iterkwargs...)

    if withsave
        folder = simpaths(model)
        savefolder = joinpath(outdir, folder, "terminal", addpath)
        if !isdir(savefolder) mkpath(savefolder) end
        
        filename = makefilename(model)
        savepath = joinpath(savefolder, filename)

        if isfile(savepath)
            if overwrite
                (verbose ≥ 1) && @warn "Removing file $savepath.\n"

                rm(savepath)
            else
                (verbose ≥ 1) && @warn "File $savepath already exists. If you want to overwrite it pass overwrite = true. Will copy the results into `F` and `policy`.\n"

                F̄, policy, G = loadterminal(model; outdir, addpath)

                return F̄, policy, G
            end
        end

    end

    F̄, policy, _ = vfi(model, G; verbose, iterkwargs...)
    
    if withsave
        (verbose ≥ 1) && println("Saving solution into $savepath...")
        jldsave(savepath; F̄, policy, G)
    end

    return F̄, policy, G
end