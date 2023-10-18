# Terminal functions
function terminalobjectivefunctional(χ, Xᵢ, Vᵢ, ∇Vᵢ)
    objectivefunction(χ, γᵇ(economy.t₁), economy.t₁, Xᵢ, Vᵢ, ∇Vᵢ)
end

function optimalterminalpolicy(Xᵢ, Vᵢ, ∇Vᵢ, Γ)
    objective = -Inf32
    χᵒ = 0f0;

    for χ ∈ Γ[1]
        objᵢ = terminalobjectivefunctional(χ, Xᵢ, Vᵢ, ∇Vᵢ);
        if objᵢ > objective
            χᵒ = χ
            objective = objᵢ
        end
    end

    return χᵒ
end

"""
Computes the optimal policy χᵒ over the state space X assuming αᵒ = γᵇ(t₁).
"""
function terminalpolicyovergrid(X::VectorGrid, V::FieldGrid, ∇V::VectorGrid, Γ::ActionRegularGrid)
    terminalpolicyovergrid!(similar(V), X, V, ∇V, Γ)
end
function terminalpolicyovergrid!(terminalpolicy::FieldGrid,
    X::VectorGrid, V::FieldGrid, ∇V::VectorGrid, Γ::ActionRegularGrid)
    
    @inbounds Threads.@threads for idx ∈ CartesianIndices(V)
        Xᵢ = @view X[idx, :]
        ∇Vᵢ = @view ∇V[idx, :]
        Vᵢ = @view V[idx]

        terminalpolicy[idx] = optimalterminalpolicy(Xᵢ, Vᵢ, ∇Vᵢ, Γ)
    end

    return terminalpolicy
end

function hjbterminal(χ, Xᵢ, Vᵢ, ∇Vᵢ, ∂²Vᵢ)
    hjb(χ, γᵇ(economy.t₁), economy.t₁, Xᵢ, Vᵢ, ∇Vᵢ, ∂²Vᵢ)
end

"""
Computes the terminal hjb over the grid X given a steady state value function V̄, Ḡ(V̄, X).
"""
Ḡ(V̄::FieldGrid, X::VectorGrid, Ω::StateRegularGrid, Γ::ActionRegularGrid) = Ḡ!(similar(V̄), V̄, X, Ω, Γ)
function Ḡ!(∂ₜV::FieldGrid, V̄::FieldGrid, X::VectorGrid, Ω::StateRegularGrid, Γ::ActionRegularGrid) # FIXME: A bit ugly.
    χ̄ = terminalpolicyovergrid(X, V̄, central∇(V̄, Ω), Γ)
    w = drift(economy.t₁, X, γᵇ(economy.t₁), χ̄);
    ∇V = dir∇(V̄, w, Ω);
    ∂ₜV .= f.(χ̄, X[:, :, :, 3], V̄, Ref(economy)) + ∇V[:, :, :, 4] .+ ∂²(1, V̄, Ω) .* hogg.σ²ₜ / 2f0
end