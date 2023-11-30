function ∂TVᵢ(Xᵢ::Point, σ²ₜ, VᵢT₊, VᵢT₋, model::ModelInstance)
    μᵢ = μ(Xᵢ.T, Xᵢ.m, model.hogg, model.albedo) * model.grid.h / model.grid.Δ[1]

    VᵢT = ifelse(μᵢ > 0, VᵢT₊, VᵢT₋)

    return (σ²ₜ / 2.) * (VᵢT₊ + VᵢT₋) + abs(μᵢ) * VᵢT
end

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

function terminaljacobi!(V̄::Array{Float64, 3}, policy::Array{Float64, 3}, model::ModelInstance)
    @unpack grid, economy, hogg, albedo = model

    σ̃ₜ² = (hogg.σₜ / (hogg.ϵ * grid.Δ[:T]))^2
    σ̃ₖ² = (economy.σₖ / grid.Δ[:y])^2

    ∑σ² = σ̃ₜ² + σ̃ₖ²

    indices = CartesianIndices(grid)
    L, R = extrema(indices)

    let Vᵢ = 0.
    for idx in CartesianIndices(grid) 
        Xᵢ = grid.X[idx]
        dT = μ(Xᵢ.T, Xᵢ.m, hogg, albedo)
        dȳ = max(abs(bterminal(Xᵢ, 0., model)), abs(bterminal(Xᵢ, 1., model)))

        Qᵢ = ∑σ² + grid.h * (abs(dT / grid.Δ[:T]) + abs(dȳ / grid.Δ[:y]))

        Vᵢ = V̄[idx]

        # Temperature
        VᵢT₊, VᵢT₋ = V̄[min(idx + I[1], R)], V̄[max(idx - I[1], L)]   
        VᵢT = ifelse(dT > 0, VᵢT₊, VᵢT₋)
        V̄[idx] = ((σ̃ₜ² / 2.) * (VᵢT₊ + VᵢT₋) + grid.h * abs(dT / grid.Δ[:T]) * VᵢT) / Qᵢ

        # GDP
        Vᵢy₊, Vᵢy₋ = V̄[min(idx + I[3], R)], V̄[max(idx - I[3], L)]
        χ, objᵪ = optimalterminalpolicy(Xᵢ, Vᵢ, Vᵢy₊, Vᵢy₋, model)

        V̄[idx] += ((σ̃ₖ² / 2.) * (Vᵢy₊ + Vᵢy₋) + grid.h * objᵪ) / Qᵢ

        # Probability of remaining in the same state
        p = ∑σ² + grid.h * (abs(dT / grid.Δ[:T]) + abs(bterminal(Xᵢ, χ, model) / grid.Δ[:y]))
        V̄[idx] += Vᵢ * (1 - p / Qᵢ)

        # Policy
        policy[idx] = χ
    end end
end