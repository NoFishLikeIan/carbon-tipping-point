using Model
using JLD2, DotEnv
using UnPack: @unpack
using Polyester: @batch

const env = DotEnv.config()
const DATAPATH = get(env, "DATAPATH", "data/")

"Computes the Jacobi iteration for the terminal problem, V̄."
function terminaljacobi!(V̄::AbstractArray{Float64, 3}, policy::AbstractArray{Float64, 3}, model::ModelInstance)
    @unpack grid, economy, hogg, albedo = model

    σₜ² = (hogg.σₜ / (hogg.ϵ * grid.Δ.T))^2
    σₖ² = (economy.σₖ / grid.Δ.y)^2

    ∑σ² = σₜ² + σₖ²

    indices = CartesianIndices(grid)
    L, R = extrema(indices)

    @batch for idx in indices
        Xᵢ = grid.X[idx]
        dT = μ(Xᵢ.T, Xᵢ.m, hogg, albedo) / (hogg.ϵ * grid.Δ.T)

        # Upper bounds on drift
        dT̄ = abs(dT)
        dȳ = max(
            abs(bterminal(Xᵢ, 0., model)), 
            abs(bterminal(Xᵢ, 1., model))
        ) / grid.Δ.y

        Qᵢ = ∑σ² + grid.h * (dT̄ + dȳ)

        # Neighbouring nodes
        Vᵢy₊, Vᵢy₋ = V̄[min(idx + I[3], R)], V̄[max(idx - I[3], L)]
        VᵢT₊, VᵢT₋ = V̄[min(idx + I[1], R)], V̄[max(idx - I[1], L)]

        # Optimal control
        χ = optimalterminalpolicy(Xᵢ, V̄[idx], Vᵢy₊, Vᵢy₋, model; tol = 1e-15)

        # Probabilities
        # -- GDP
        dy = bterminal(Xᵢ, χ, model) / grid.Δ.y
        Py₊ = ((σₖ² / 2.) + grid.h * max(dy, 0.)) / Qᵢ
        Py₋ = ((σₖ² / 2.) + grid.h * max(-dy, 0.)) / Qᵢ

        # -- Temperature
        PT₊ = ((σₜ² / 2.) + grid.h * max(dT, 0.)) / Qᵢ
        PT₋ = ((σₜ² / 2.) + grid.h * max(-dT, 0.)) / Qᵢ

        # -- Residual
        P = Py₊ + Py₋ + PT₊ + PT₋

        V̄[idx] = (
            PT₊ * VᵢT₊ + PT₋ * VᵢT₋ +
            Py₊ * Vᵢy₊ + Py₋ * Vᵢy₋ +
            (grid.h)^2 * f(χ, Xᵢ.y, V̄[idx], economy)
        ) / P

        policy[idx] = χ
    end

    return V̄, policy
end

relerror(x, y) = abs(x - y) / abs(y)

function terminaliteration(V₀::AbstractArray{Float64, 3}, model::ModelInstance; rtol = 1e-3, maxiter = 100_000, verbose = false)
    policy = similar(V₀)

    verbose && println("Starting iterations...")

    Vᵢ, Vᵢ₊₁ = copy(V₀), copy(V₀);
    for iter in 1:maxiter
        terminaljacobi!(Vᵢ₊₁, policy, model)
        ε = maximum(relerror.(Vᵢ₊₁, Vᵢ))
        if ε < rtol
            return Vᵢ₊₁, policy, model
        end
        
        Vᵢ .= Vᵢ₊₁
        verbose && print("Iteration $iter / $maxiter, ε = $ε...\r")
    end

    verbose && println("\nDone without convergence.")
    return Vᵢ₊₁, policy, model
end

function computeterminal(N::Int, Δλ = 0.08; 
    consapprox = [10, 4, 2], rtol = 1e-4, 
    verbose = true, withsave = true, 
    iterkwargs...)

    calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
    hogg = Hogg()
    economy = Economy()
    albedo = Albedo(λ₂ = Albedo().λ₁ - Δλ)

    domains = [
        (hogg.Tᵖ, hogg.Tᵖ + 10.), 
        (log(hogg.M₀), Model.mstable(hogg.Tᵖ + 10., hogg, albedo)), 
        (log(economy.Y₀ / 2), log(economy.Y₀ * 2))
    ]

    h = 1 / N

    verbose && println("Constructing initial condition...")
    hs = consapprox .* h
    rtols = rtol ./ consapprox
    
    gⁱ⁻¹ = RegularGrid(domains, first(hs))
    V̄ = -h * ones(size(gⁱ⁻¹))

    for (i, hⁱ) in enumerate(hs)
        verbose && println("\nApproximation $i / $(length(consapprox)) with hⁱ = $hⁱ")

        gⁱ = RegularGrid(domains, hⁱ)
        V₀ = interpolateovergrid(gⁱ⁻¹, V̄, gⁱ)

        modelⁱ = ModelInstance(economy, hogg, albedo, gⁱ, calibration)

        V̄ = first(terminaliteration(V₀, modelⁱ; verbose, rtol = rtols[i], iterkwargs...))
        gⁱ⁻¹ = gⁱ
    end

    verbose && println("Computing with h = $h...")
    grid = RegularGrid(domains, h)
    model = ModelInstance(economy, hogg, albedo, grid, calibration)

    V₀ = interpolateovergrid(gⁱ⁻¹, V̄, grid)
    V̄, policy, model = terminaliteration(V₀, model; verbose, rtol = rtol, iterkwargs...)

    if withsave
        savepath = joinpath(DATAPATH, "terminal", "N=$(N)_Δλ=$(Δλ).jld2")
        println("\nSaving solution into $savepath...")

        jldsave(savepath; V̄, policy, model)
    end

    return V̄
end