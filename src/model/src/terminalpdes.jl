"""
Computes the terminal Hamilton-Jacobi-Bellmann equation at point Xᵢ
"""
function hjbterminal(χᵢ, Xᵢ, Vᵢ, ∂yVᵢ, ∂²Vᵢ)
    f(χᵢ, Xᵢ[2], Vᵢ[1], economy) + 
        ∂yVᵢ[1] * (
            ϕ(economy.t₁, χᵢ, economy) - δₖ(Xᵢ[1], economy, hogg)
        ) + ∂²Vᵢ[1] * hogg.σ²ₜ / 2f0
end

function terfoc(χ, Xᵢ, Vᵢ, ∂yVᵢ)
    t = economy.t₁
    Y∂f(χ, Xᵢ[2], Vᵢ[1], economy) + ∂yVᵢ[1] * ϕ′(t, χ, economy)
end

function optimalterminalpolicy(Xᵢ, Vᵢ, ∂yVᵢ)
    g(χ) = terfoc(χ, Xᵢ, Vᵢ, ∂yVᵢ)

    if g(1f-3) * g(1f0) > 0 
        if g(1f-3) < 0 return 1f-3 end
        if g(1f0) > 0 return 1f0 end
    end

    find_zero(g, (1f-3, 1f0), Bisection())
end

"""
Computes the optimal policy χ' over the state space X
"""
function terminalpolicyovergrid(X, V, ∂yV)
    terminalpolicyovergrid!(similar(V), X, V, ∂yV)
end
function terminalpolicyovergrid!(policy, X, V, ∂yV)
    @batch for idx ∈ CartesianIndices(V)
        Xᵢ = @view X[idx, :]
        ∂yVᵢ = @view ∂yV[idx]
        Vᵢ = @view V[idx]

        policy[idx] = optimalterminalpolicy(Xᵢ, Vᵢ, ∂yVᵢ)
    end
    
    return policy
end

function ydrift!(w, policy, T)
    w .= ϕ.(economy.t₁, policy, Ref(economy)) .- δₖ.(T, Ref(economy), Ref(hogg))
end

function terminalG(X, V, Ω)
    terminalG!(
        similar(V), similar(V, size(V)..., 4),
        X, V, Ω
    )
end
"""
Computes G! by modifying ∂ₜV and tmp = (∂yV, ∂²TV,  policy, w)
"""
function terminalG!(∂ₜV, tmp, X, V, Ω)
    ∂yV = @view tmp[:, :, 1]
    ∂²TV = @view tmp[:, :, 2]
    policy = @view tmp[:, :, 3]
    w = @view tmp[:, :, 4]
    T = @view X[:, :, 1]
    y = @view X[:, :, 2]

    central∂!(∂yV, V, Ω; direction = 2);
    terminalpolicyovergrid!(policy, X, V, ∂yV);
    ydrift!(w, policy, T)
    dir∂!(∂yV, V, w, Ω; direction = 2);
    ∂²!(∂²TV, V, Ω; dim = 1)


    for idx in CartesianIndices(∂ₜV)
        ∂ₜV[idx] = hjbterminal(policy[idx], X[idx, :], V[idx], ∂yV[idx], ∂²TV[idx])
    end

    return ∂ₜV
end