function jacobi!(Vₜ::Array{Float64, 3}, policy::AbstractArray{Policy, 3}, t, model::ModelInstance)
    @unpack grid, economy, hogg, albedo, calibration = model

    σ̃ₜ² = (hogg.σₜ / (hogg.ϵ * grid.Δ.T))^2
    σ̃ₖ² = (economy.σₖ / grid.Δ.y)^2

    ∑σ² = σ̃ₜ² + σ̃ₖ²

    γₜ = γ(t, economy, calibration)
    dm̄ = γₜ / grid.Δ.m

    indices = CartesianIndices(grid)
    L, R = extrema(indices)
    U = oneunit(L)

    @batch for idx in indices
        Xᵢ = grid.X[idx]
        dT = μ(Xᵢ.T, Xᵢ.m, hogg, albedo) / (hogg.ϵ * grid.Δ.T)

        # Upper bounds on drift
        dT̄ = abs(dT) 
        dȳ = max(abs(bterminal(Xᵢ, 0., model)), abs(bterminal(Xᵢ, 1., model))) / grid.Δ.y

        Qᵢ = ∑σ² + grid.h * (dT̄ + dȳ + dm̄)

        # Neighbouring nodes
        cube = (max(idx - U, L)):(min(idx + U, R))

        VᵢT₊, VᵢT₋ = Vₜ[min(idx + I[1], R)], Vₜ[max(idx - I[1], L)]
        Vᵢm₊ = Vₜ[min(idx + I[2], R)]
        Vᵢy₊, Vᵢy₋ = Vₜ[min(idx + I[3], R)], Vₜ[max(idx - I[3], L)]

        # Optimal control
        policyᵢ = optimalpolicy(
            t, Xᵢ, Vₜ[idx], Vᵢy₊, Vᵢm₊, Vᵢy₋, model; 
            policy₀ = Vector(mean(policy[cube]))
        )

        # Probabilities
        # -- Temperature
        PT₊ = ((σ̃ₜ² / 2.) + grid.h * max(dT, 0.)) / Qᵢ
        PT₋ = ((σ̃ₜ² / 2.) + grid.h * max(-dT, 0.)) / Qᵢ

        # -- Carbon concentration
        dm = γₜ - policyᵢ.α
        Pm₊ = grid.h * dm / Qᵢ

        # -- GDP
        dy = b(t, Xᵢ, policyᵢ, model) / grid.Δ.y
        Py₊ = ((σ̃ₖ² / 2.) + grid.h * max(dy, 0.)) / Qᵢ
        Py₋ = ((σ̃ₖ² / 2.) + grid.h * max(-dy, 0.)) / Qᵢ

        # -- Residual
        P = Py₊ + Py₋ + PT₊ + PT₋ + Pm₊

        Vₜ[idx] = (
            PT₊ * VᵢT₊ + PT₋ * VᵢT₋ +
            Py₊ * Vᵢy₊ + Py₋ * Vᵢy₋ +
            Pm₊ * Vᵢm₊ +
            (grid.h)^2 * f(policyᵢ.χ, Xᵢ.y, Vₜ[idx], economy)
        ) / P

        policy[idx] = policyᵢ
    end
end

"Computes the Jacobi iteration for the terminal problem, V̄."
function terminaljacobi!(V̄::AbstractArray{Float64, 3}, policy::AbstractArray{Float64, 3}, model::ModelInstance)
    @unpack grid, economy, hogg, albedo = model

    σ̃ₜ² = (hogg.σₜ / (hogg.ϵ * grid.Δ.T))^2
    σ̃ₖ² = (economy.σₖ / grid.Δ.y)^2

    ∑σ² = σ̃ₜ² + σ̃ₖ²

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
        χ = optimalterminalpolicy(Xᵢ, V̄[idx], Vᵢy₊, Vᵢy₋, model)

        # Probabilities
        # -- GDP
        dy = bterminal(Xᵢ, χ, model) / grid.Δ.y
        Py₊ = ((σ̃ₖ² / 2.) + grid.h * max(dy, 0.)) / Qᵢ
        Py₋ = ((σ̃ₖ² / 2.) + grid.h * max(-dy, 0.)) / Qᵢ

        # -- Temperature
        PT₊ = ((σ̃ₜ² / 2.) + grid.h * max(dT, 0.)) / Qᵢ
        PT₋ = ((σ̃ₜ² / 2.) + grid.h * max(-dT, 0.)) / Qᵢ

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