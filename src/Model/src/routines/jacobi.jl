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
    @unpack grid, economy = model

    σ̃²ₜ, σ̃²ₖ = σ̃²(model)
    ∑σ² = σ̃²ₜ + σ̃²ₖ 
    indices = CartesianIndices(grid)
    L, R = extrema(indices)

    let Vᵢ = 0.
    for idx in CartesianIndices(grid) 
        Qᵢ = ∑σ² + grid.h * sum(boundterminaldrift(grid.X[idx], model))
        Vᵢ = V̄[idx]
        VᵢT₊, VᵢT₋ = V̄[min(idx + I[1], R)], V̄[max(idx - I[1], L)]
        Vᵢy₊, Vᵢy₋ = V̄[min(idx + I[3], R)], V̄[max(idx - I[3], L)]

        χ, objᵪ = optimalterminalpolicy(grid.X[idx], Vᵢ, Vᵢy₊, Vᵢy₋, model)

        policy[idx] = χ
        V̄[idx] = ( 
            ∂TVᵢ(grid.X[idx], σ̃²ₜ, VᵢT₊, VᵢT₋, model) + # Temperature component
            (σ̃²ₖ / 2.) * (Vᵢy₊ + Vᵢy₋) + objᵪ * grid.h  # GDP component
        ) / Qᵢ

        p = (∑σ² + grid.h * sum(terminaldrift(grid.X[idx], χ, model))) / Qᵢ

        V̄[idx] += Vᵢ * (1 - p)
    end end
end