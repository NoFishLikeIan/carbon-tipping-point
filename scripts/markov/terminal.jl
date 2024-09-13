using Distributed: @everywhere, @distributed, @sync
using SharedArrays

@everywhere begin
    using Model, Grid
    using JLD2, DotEnv

    using FastClosures: @closure
    using Printf: @printf, @sprintf
end

@everywhere function terminalcost(Fᵢ′, Tᵢ, Δt, χ, model::AbstractModel{GrowthDamages, P}) where P
    δ = terminaloutputfct(Tᵢ, Δt, χ, model)

    g(χ, δ * Fᵢ′, Δt, model.preferences)
end
@everywhere function terminalcost(Fᵢ′, Tᵢ, Δt, χ, model::AbstractModel{LevelDamages, P}) where P
    δ = terminaloutputfct(Tᵢ, Δt, χ, model)
    damage = d(Tᵢ, model.damages, model.hogg)

    g(damage * χ, δ * Fᵢ′, Δt, model.preferences)
end

@everywhere function terminaldriftstep(idx, F̄, model::AbstractModel, G)
    L, R = extrema(CartesianIndices(F̄))

    σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * G.Δ.T))^2
    σₘ² = (model.hogg.σₘ / G.Δ.m)^2

    Xᵢ = G.X[idx]
    dT = μ(Xᵢ.T, Xᵢ.m, model) / (model.hogg.ϵ * G.Δ.T)

    # Neighbouring nodes
    FᵢT₊, FᵢT₋ = F̄[min(idx + I[1], R)], F̄[max(idx - I[1], L)]
    Fᵢm₊, Fᵢm₋ = F̄[min(idx + I[2], R)], F̄[max(idx - I[2], L)]

    dTF = G.h * abs(dT) * ifelse(dT > 0, FᵢT₊, FᵢT₋) + σₜ² * (FᵢT₊ + FᵢT₋) / 2
    dmF = σₘ² * (Fᵢm₊ + Fᵢm₋) / 2

    Q = σₘ² + σₜ² + G.h * abs(dT)
    F′ = (dTF + dmF) / Q
    Δt = G.h^2 / Q

    return F′, Δt
end

@everywhere terminalmarkovstep(idx, F̄, model::TippingModel, G) = terminaldriftstep(idx, F̄, model, G)
@everywhere function terminalmarkovstep(idx, F̄, model::JumpModel, G)
    Fᵈ, Δt = terminaldriftstep(idx, F̄, model, G)

    # Update with jump
    R = maximum(CartesianIndices(F̄))
    Xᵢ = G.X[idx]
    πᵢ = intensity(Xᵢ.T, model.hogg, model.jump)
    qᵢ = increase(Xᵢ.T, model.hogg, model.jump)

    steps = floor(Int, div(qᵢ, G.Δ.T * G.h))
    weight = qᵢ / (G.Δ.T * G.h)

    Fʲ = F̄[min(idx + steps * I[1], R)] * (1 - weight) + 
            F̄[min(idx + (steps + 1) * I[1], R)] * weight

    F′ = Fᵈ + πᵢ * Δt * (Fʲ - Fᵈ)

    return F′, Δt
end

function terminaljacobi!(F̄, policy, errors, model::AbstractModel, G; indices = CartesianIndices(F̄))

    @sync @distributed for idx in indices
        Fᵢ′, Δt = terminalmarkovstep(idx, F̄, model, G)
        Tᵢ = G.X[idx].T

        # Optimal control
        objective = @closure χ -> terminalcost(Fᵢ′, Tᵢ, Δt, χ, model)
        Fᵢ, χ = gssmin(objective, 0., 1.; tol = eps(Float64))
        
        errors[idx] = abs(Fᵢ - F̄[idx]) / F̄[idx]

        F̄[idx] = Fᵢ
        policy[idx] = χ
    end

end

function vfi(F₀, model::AbstractModel, G; tol = 1e-3, maxiter = 10_000, verbose = false, indices = CartesianIndices(G), alternate = false)
    Fᵢ = deepcopy(F₀) |> SharedMatrix
    pᵢ = similar(F₀) |> SharedMatrix

    errors = (Inf .* ones(size(F₀))) |> SharedMatrix
    
    verbose && println("Starting iterations...")

    for iter in 1:maxiter
        iterindices = (alternate && isodd(iter)) ? indices : reverse(indices)

        terminaljacobi!(Fᵢ, pᵢ, errors, model, G; indices = iterindices)

        max_error = maximum(errors)

        if max_error < tol
            verbose && @printf("Converged in %i iterations, ε = %.8f \n", iter, max_error)
            return Fᵢ, pᵢ
        end

        if verbose && (!alternate || isodd(iter))
            @printf("Iteration %i / %i, ε = %.8f...\r", iter, maxiter, max_error)
        end
    end

    verbose && @warn "Convergence failed."
    return Fᵢ, pᵢ
end

function computeterminal(model, G::RegularGrid; verbose = true, withsave = true, datapath = "data", overwrite = false, addpath = "", iterkwargs...)

    if withsave
        folder = SIMPATHS[typeof(model)]
        savefolder = joinpath(datapath, folder, "terminal", addpath)
        if !isdir(savefolder) mkpath(savefolder) end
        
        filename = makefilename(model)
        savepath = joinpath(savefolder, filename)

        if isfile(savepath)
            if overwrite
                verbose && @warn "Removing file $savepath.\n"

                rm(savepath)
            else
                verbose && @warn "File $savepath already exists. If you want to overwrite it pass overwrite = true. Will copy the results into `F` and `policy`.\n"

                F̄, policy, G = loadterminal(model; datapath, addpath)

                return F̄, policy, G
            end
        end

    end

    F₀ = ones(size(G))
    F̄, policy = vfi(F₀, model, G; verbose, iterkwargs...)
    
    if withsave
        verbose && println("Saving solution into $savepath...")
        jldsave(savepath; F̄, policy, G)
    end

    return F̄, policy, G
end