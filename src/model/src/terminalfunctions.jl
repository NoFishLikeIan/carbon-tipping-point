"Computes the drift of `y` in the post transition phase."
function ȳdrift!(w, X::AbstractArray, χ::AbstractArray, instance::ModelInstance)
    economy, hogg, _ = instance

    for idx in CartesianIndices(w)        
        w[idx] = ϕ(economy.τ, χ[idx], economy) - economy.δₖᵖ - (hogg.σ²ₜ / 2f0) * d′′(X[idx, 1], economy, hogg)         
    end
end

"""
Computes the terminal first order condition of the univariate method, 
    ∂ᵪf(χ, y, V) + ϕ′(χ) ∂yV = 0
"""
function terminalfoc(χ, yᵢ::Real, Vᵢ::Real, ∂yVᵢ::Real, economy::Economy)
    Y∂f(χ, yᵢ, Vᵢ, economy) + ϕ′(economy.τ, χ, economy) * ∂yVᵢ
end
terminalfoc(χ, yᵢ::AbstractArray, Vᵢ::AbstractArray, ∂yVᵢ::AbstractArray, economy::Economy) = terminalfoc(χ, first(yᵢ), first(Vᵢ), first(∂yVᵢ), economy)

function hjbterminal(χᵢ, Xᵢ, Vᵢ, ∂yVᵢ, ∂²yVᵢ, instance::ModelInstance)
    economy, hogg, _ = instance

    f(χᵢ, Xᵢ[2], Vᵢ[1], economy) + 
        ∂yVᵢ[1] * ϕ(economy.τ, χᵢ, economy) + 
        (hogg.σ²ₜ / 2f0) * (
            ∂²yVᵢ[1] - ∂yVᵢ[1] * d′′(Xᵢ[1], economy, hogg)
        )

end