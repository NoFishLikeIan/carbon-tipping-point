"Computes the drift of `y` in the post transition phase."
function ȳdrift!(ẏ, χ::AbstractArray, grid::RegularGrid, instance::ModelInstance)
    economy, hogg, _ = instance

    @batch for idx in CartesianIndices(grid) 
        ẏ[idx] = ϕ(economy.τ, χ[idx], economy) - economy.δₖᵖ - d(grid.X[idx, 1], economy, hogg)
    end
end

"""
Computes the terminal first order condition of the univariate method, 
    ∂ᵪf(χ, y, V) + ϕ′(χ) ∂yV = 0
"""
function terminalfoc(χᵢ, Xᵢ, Vᵢ::Real, ∂V∂yᵢ::Real, economy::Economy)
    y = Xᵢ[3]
    Y∂f(χᵢ, y, Vᵢ, economy) + ϕ′(economy.τ, χᵢ, economy) * ∂V∂yᵢ
end

function hjbterminal(χᵢ, Xᵢ, Vᵢ::Real, ∂V∂Tᵢ::Real, ∂V∂yᵢ::Real, ∂²V∂T²ᵢ::Real, instance::ModelInstance)
    economy, hogg, albedo = instance
    Tᵢ, mᵢ, yᵢ = Xᵢ

    f(χᵢ, yᵢ, Vᵢ, economy) + 
        ∂V∂yᵢ * (
            ϕ(economy.τ, χᵢ, economy) - economy.δₖᵖ - d(Tᵢ, economy, hogg)
        ) +
        ∂V∂Tᵢ * Model.μ(Tᵢ, mᵢ, hogg, albedo) / hogg.ϵ +
        ∂²V∂T²ᵢ * hogg.σ²ₜ / 2f0hogg.ϵ^2
end