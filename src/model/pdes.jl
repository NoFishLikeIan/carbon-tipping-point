function ε(t, M, α)
    1f0 .- M .* (δₘ.(M, Ref(hogg)) .+ γᵇ(t) .- α) ./ (Gtonoverppm * Eᵇ(t))
end

function drift(t, X, α, χ)
    T = @view X[:, :, :, 1]
    m = @view X[:, :, :, 2]

    w = similar(X)

    w[:, :, :, 1] .= μ.(T, m, Ref(hogg), Ref(albedo))

    w[:, :, :, 2] .= γᵇ.(t) .- α
    w[:, :, :, 3] .= economy.ϱ .+ ϕ.(χ, A(t, economy), Ref(economy)) .- A(t, economy) .* β.(t, ε(t, exp.(m), α), Ref(economy)) .- δₖ.(T, Ref(economy), Ref(hogg))

    return w
end

function objectivefunction(χ, α, t, Xᵢ, Vᵢ, ∇Vᵢ)
    f(χ, Xᵢ[3], Vᵢ[1], economy) + 
        ∇Vᵢ[2] * (γᵇ(t) - α) + 
        ∇Vᵢ[3] * (
            ϕ(χ,  A(t, economy), economy) -  A(t, economy) * β(t, ε(t, exp(Xᵢ[2]), α), economy)
        )
end

function optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ, Γ)
    objective = -Inf32
    χᵒ = 0f0;
    αᵒ = 0f0; 

    for χ ∈ Γ[1], α ∈ Γ[2]
        objᵢ = objectivefunction(χ, α, t, Xᵢ, Vᵢ, ∇Vᵢ);
        if objᵢ > objective
            χᵒ = χ
            αᵒ = α
            objective = objᵢ
        end
    end

    return χᵒ, αᵒ
end

"""
Computes the optimal policy (χᵒ, αᵒ) over the state space X
"""
function policyovergrid(t::Float32, X::VectorGrid, V::FieldGrid, ∇V::VectorGrid, Γ)
    policyovergrid!(
        Array{Float32}(undef, size(V)..., 2), 
        t, X, V, ∇V, Γ
    )
end
function policyovergrid!(
    policy::SharedVectorGrid, 
    t::Float32, X::VectorGrid, V::FieldGrid, ∇V::VectorGrid, Γ)
    
    @inbounds @distributed for idx ∈ CartesianIndices(V)
        Xᵢ = @view X[idx, :]
        ∇Vᵢ = @view ∇V[idx, :]
        Vᵢ = @view V[idx]

        policy[idx, :] .= optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ, Γ)
    end

    return policy
end
function policyovergrid!(
    policy::VectorGrid, 
    t::Float32, X::VectorGrid, V::FieldGrid, ∇V::VectorGrid, Γ)
    
    @inbounds for idx ∈ CartesianIndices(V)
        Xᵢ = @view X[idx, :]
        ∇Vᵢ = @view ∇V[idx, :]
        Vᵢ = @view V[idx]

        policy[idx, :] .= optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ, Γ)
    end

    return policy
end

function hjb(χ, α, t, Xᵢ, Vᵢ, ∇Vᵢ, ∂²Vᵢ)
    f(χ, Xᵢ[3], Vᵢ[1], economy) + 
        ∇Vᵢ[1] * μ(Xᵢ[1], Xᵢ[2], hogg, albedo) +
        ∇Vᵢ[2] * (γᵇ(t) - α) + 
        ∇Vᵢ[3] * (
            economy.ϱ + ϕ(χ,  A(t, economy), economy) - 
            A(t, economy) * β(t, ε(t, exp(Xᵢ[2]), α), economy) -
            δₖ(Xᵢ[1], economy, hogg)
        ) +
        ∂²Vᵢ[1] * hogg.σ²ₜ / 2f0
end

G(t::Float32, X::VectorGrid, V::FieldGrid, ∇V::VectorGrid, policy::VectorGrid) = G!(similar(V), t, X, V, ∇V, policy)
function G!(∂ₜV::FieldGrid, t::Float32, X::VectorGrid, V::FieldGrid, Ω, Γ)
    policy = policyovergrid(t, X, V, central∇(V, Ω), Γ);
    w = drift(economy.t₁, X, γᵇ(economy.t₁), χ̄);
    ∇V = dir∇(V̄, w, Ω);
    ∂ₜV .= f.(χ̄, X[:, :, :, 3], V̄, Ref(economy)) + ∇V[:, :, :, 4] .+ ∂²(1, V̄, Ω) .* hogg.σ²ₜ / 2f0
end