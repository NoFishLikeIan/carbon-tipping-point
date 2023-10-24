using Polyester: @batch
using Roots: Bisection, find_zero

"""
Computes the terminal Hamilton-Jacobi-Bellmann equation at point Xᵢ
"""
function hjbterminal(χ, Xᵢ, Vᵢ, ∂yVᵢ, ∂²Vᵢ)
    t = economy.t₁

    f(χ, Xᵢ[2], Vᵢ[1], economy) + 
        ∂yVᵢ[1] * (ϕ(t, χ, economy) - δₖ(Xᵢ[1], economy, hogg)) +
        ∂²Vᵢ[1] * hogg.σ²ₜ / 2f0
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

function terminalG(X, V, Ω)
    terminalG!(
        similar(V), similar(V), similar(V), similar(V),
        X, V, Ω
    )
end
"""
Computes G! by modifying (∂ₜV, ∂yV, policy, w)
"""
function terminalG!(∂ₜV, ∂yV, w, policy, X, V, Ω)
    central∂!(∂yV, V, Ω; direction = 2);
    terminalpolicyovergrid!(policy, X, V, ∂yV);

    T = @view X[:, :, 1]
    y = @view X[:, :, 2]

    w .= ϕ.(economy.t₁, policy, Ref(economy)) .- δₖ.(T, Ref(economy), Ref(hogg))
    
    dir∂!(∂yV, V, w, Ω; direction = 2);

    ∂ₜV .= f.(policy, y, V, Ref(economy)) + (∂yV .* w) .+ ∂²(V, Ω; dim = 1) .* hogg.σ²ₜ / 2f0

    return ∂ₜV
end