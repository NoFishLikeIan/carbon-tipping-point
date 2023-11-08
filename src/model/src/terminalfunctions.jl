"Computes the drift of `y` in the post transition phase."
function ȳdrift!(w, X::AbstractArray, χ::AbstractArray, instance::ModelInstance)
    economy, hogg, albedo = instance

    @batch for idx in CartesianIndices(w)
        T = @view X[idx, 1]
        
        w[idx] = ϕ(economy.τ, χ, economy) - economy.δₖᵖ + d′(T, economy, hogg) * μ(T, m, hogg, albedo) + (hogg.σ²ₜ / 2f0) * d′′(T, economy, hogg)
                
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