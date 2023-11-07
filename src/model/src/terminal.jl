"Computes the drift of `y` in the post transition phase."
function ȳdrift!(w, X::AbstractArray, χ::AbstractArray, instance::ModelInstance)
    economy, hogg, albedo = instance

    @batch for idx in CartesianIndices(w)
        T = @view X[idx, 1]
        
        w[idx] = ϕ(economy.τ, χ, economy) - economy.δₖᵖ + d′(T, economy, hogg) * μ(T, m, hogg, albedo) + (hogg.σ²ₜ / 2f0) * d′′(T, economy, hogg)
                
    end
end

function objective!(z, ∂ᵪ, ∂²ᵪ, χ, Xᵢ, Vᵢ, ∂₁V, instance::ModelInstance, calibration::Calibration)

    economy = first(instance)
    f₀, Yf₁, Y²f₂ = epsteinzinsystem(χ, Xᵢ[3], Vᵢ[1], economy)

    # ... ? 
end