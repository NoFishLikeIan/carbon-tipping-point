function jacobi!(Vₜ::Array{Float64, 3}, t, model::ModelInstance)
    @unpack grid, economy = model

    let pᵢ = 0., Vᵢ = 0., σ² = var(model)
    @batch for idx in CartesianIndices(grid)
        Vᵢ = Vₜ[idx]
        Vᵢy₊ = V[idx + I[3]]
        Vᵢy₋ = V[idx - I[3]]
        Vᵢm₊ = V[idx + I[2]]

        policy, minimizer = optimalpolicy(t, Xᵢ, Vᵢ, Vᵢy₊, Vᵢy₋, Vᵢm₊, model)
        
        # Finish...

        Vₜ[idx] += (1.0 - pᵢ) * Vᵢ
        Vₜ[idx] /= Q(t, Xᵢ, model)
    end end
end

"""
Computes the Jacobi iteration for the terminal problem, V̄. 
"""
function terminaljacobi!(V̄::AbstractArray{Float64, 3}, policy::AbstractArray{Float64, 3}, model::ModelInstance; fwdpass = true)
    @unpack grid, economy, hogg, albedo = model

    σ̃ₜ² = (hogg.σₜ / (hogg.ϵ * grid.Δ.T))^2
    σ̃ₖ² = (economy.σₖ / grid.Δ.y)^2

    ∑σ² = σ̃ₜ² + σ̃ₖ²

    indices = CartesianIndices(grid)
    L, R = extrema(indices)

    @batch for idx in (fwdpass ? indices : reverse(indices)) 
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