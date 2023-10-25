"Computes the terminal Hamilton-Jacobi-Bellmann equation at point Xᵢ"
function hjbterminal(χᵢ, Xᵢ, Vᵢ, ∂yVᵢ, ∂²Vᵢ, instance::ModelInstance)
    economy, hogg, _ = instance
    t = economy.t₁

    f(χᵢ, Xᵢ[2], Vᵢ[1], economy) + 
        ∂yVᵢ[1] * (
            ϕ(t, χᵢ, economy) - δₖ(Xᵢ[1], economy, hogg)
        ) +
        ∂²Vᵢ[1] * hogg.σ²ₜ / 2f0
end

function terfoc(χᵢ, Xᵢ, Vᵢ, ∂yVᵢ, instance::ModelInstance)
    economy = first(instance)
    t = economy.t₁
    Y∂f(χᵢ, Xᵢ[2], Vᵢ[1], economy) + ∂yVᵢ[1] * ϕ′(t, χᵢ, economy)
end

function optimalterminalpolicy(Xᵢ, Vᵢ, ∂yVᵢ, instance::ModelInstance)
    g = @closure χ -> terfoc(χ, Xᵢ, Vᵢ, ∂yVᵢ, instance)

    if g(1f-3) * g(1f0) > 0 
        if g(1f-3) < 0 return 1f-3 end
        if g(1f0) > 0 return 1f0 end
    end

    find_zero(g, (1f-3, 1f0), Bisection())
end


function ydrift!(w, policy, T, instance::ModelInstance)
    economy, hogg, _ = instance
    t = economy.t₁

    @batch for idx in CartesianIndices(w)
        w[idx] = ϕ(t, policy[idx], economy) - δₖ(T[idx], economy, hogg)
    end

    return w
end

"""
Computes the optimal policy χ' over the state space X
"""
function terminalpolicyovergrid(X, V, ∂yV, instance::ModelInstance)
    terminalpolicyovergrid!(similar(V), X, V, ∂yV, instance)
end
function terminalpolicyovergrid!(policy, X, V, ∂yV, instance::ModelInstance)
    @batch for idx ∈ CartesianIndices(V)
        Xᵢ = @view X[idx, :]
        ∂yVᵢ = @view ∂yV[idx]
        Vᵢ = @view V[idx]

        policy[idx] = optimalterminalpolicy(Xᵢ, Vᵢ, ∂yVᵢ, instance)
    end
    
    return policy
end