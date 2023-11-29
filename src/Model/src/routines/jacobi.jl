function jacobi!(Vₜ::Array{Float64, 3}, t, policy::Array{Policy, 3}, model::ModelInstance)
    @unpack grid, economy = model

    let pᵢ = 0., Vᵢ = 0., σ² = var(model)
    @batch for idx in CartesianIndices(grid)
        Vᵢ = Vₜ[idx]
        pᵢ = 0.0
        Vₜ[idx] = f(policy[idx].χ , grid.X[idx].y, Vᵢ, economy) * grid.h^2

        bᵢ = drift(t, grid.X[idx], policy[idx], model) ./ model.grid.Δ

        for (j, Iⱼ) ∈ enumerate(I)
            p₊ = (grid.h * max(bᵢ[j], 0.) + σ²[j] / 2.0)
            p₋ = (grid.h * max(-bᵢ[j], 0.) + σ²[j] / 2.0) 
            pᵢ += p₊ + p₋

            Vₜ[idx] += Vₜ[idx + Iⱼ] * p₊ + Vₜ[idx - Iⱼ] * p₋
        end

        Vₜ[idx] += (1.0 - pᵢ) * Vᵢ
        Vₜ[idx] /= Q(t, Xᵢ, model)
    end end
end