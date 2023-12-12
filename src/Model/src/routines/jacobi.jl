"Backward simulates from V̄ = V(τ) down to V(0). Stores nothing."
function backwardsimulation!(V::AbstractArray{Float64, 3}, policy::AbstractArray{Policy, 3}, model::ModelInstance; verbose = false)
    @unpack grid, economy, hogg, albedo, calibration = model
    σₜ² = (hogg.σₜ / (hogg.ϵ * grid.Δ.T))^2
    σₖ² = (economy.σₖ / grid.Δ.y)^2

    indices = CartesianIndices(grid)
    L, R = extrema(indices)

    timequeue = PriorityQueue{typeof(L), Float64}(
        Base.Order.Reverse, indices .=> economy.τ);

    verbose && println("Starting backward simulation...")

    while !isempty(timequeue)
        idx, tᵢ = dequeue_pair!(timequeue)
        Xᵢ = grid.X[idx]
        d = driftbounds(tᵢ, Xᵢ, model)
        verbose && print("Remaining values $(length(timequeue)), time $tᵢ\r")

        Qᵢ = σₜ² + σₖ² + grid.h * sum(abs(d))
        Δtᵢ = (grid.h)^2 / Qᵢ 

        # Neighbouring nodes
        VᵢT₊, VᵢT₋ = V[min(idx + I[1], R)], V[max(idx - I[1], L)]
        Vᵢm₊ = V[min(idx + I[2], R)]
        Vᵢy₊, Vᵢy₋ = V[min(idx + I[3], R)], V[max(idx - I[3], L)]

        # Optimal control
        cube = max(idx - oneunit(L), L):min(idx + oneunit(R), R)
        policyguess = mean(policy[cube])
        optpolicy = optimalpolicy(tᵢ, Xᵢ, V[idx], Vᵢy₊, Vᵢm₊, Vᵢy₋, model; p₀ = policyguess)

        # -- Temperature
        PT₊ = ((σₜ² / 2.) + grid.h * max(d.dT, 0.)) / Qᵢ
        PT₋ = ((σₜ² / 2.) + grid.h * max(-d.dT, 0.)) / Qᵢ

        # -- Carbon concentration
        dm = d.dm - (optpolicy.α / grid.Δ.m)
        Pm₊ = grid.h * dm / Qᵢ

        # -- GDP
        dy = b(tᵢ, Xᵢ, optpolicy, model) / grid.Δ.y
        Py₊ = ((σₖ² / 2.) + grid.h * max(dy, 0.)) / Qᵢ
        Py₋ = ((σₖ² / 2.) + grid.h * max(-dy, 0.)) / Qᵢ

        # -- Residual
        P = Py₊ + Py₋ + PT₊ + PT₋ + Pm₊

        V[idx] = (
            PT₊ * VᵢT₊ + PT₋ * VᵢT₋ +
            Py₊ * Vᵢy₊ + Py₋ * Vᵢy₋ +
            Pm₊ * Vᵢm₊ ) / P +
            Δtᵢ * f(optpolicy.χ, Xᵢ.y, V[idx], economy)

        if tᵢ > 0
            push!(timequeue, idx => tᵢ - Δtᵢ)
        end        
    end

    return V, policy
end

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
        χ = optimalterminalpolicy(Xᵢ, V̄[idx], Vᵢy₊, Vᵢy₋, model)

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