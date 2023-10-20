"""
Computes the Hamilton-Jacobi-Bellmann equation at point Xᵢ
"""
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

"""
Computes the the objective functional at point Xᵢ which depends on the control.
"""
function objectivefunction(χ, α, t, Xᵢ, Vᵢ, ∇Vᵢ)
    f(χ, Xᵢ[3], Vᵢ[1], economy) + 
        ∇Vᵢ[2] * (γᵇ(t) - α) + 
        ∇Vᵢ[3] * (
            ϕ(χ,  A(t, economy), economy) -  A(t, economy) * β(t, ε(t, exp(Xᵢ[2]), α), economy)
        )
end


"""
Computes the the optimal policy at point Xᵢ. P is assumed to be a Latin Hyper Grid.
"""
function optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ, P)
    objective = -Inf32
    copt = @view P[1, :]

    for i ∈ axes(P, 1)
        c = @view P[i, :]

        o = objectivefunction(c[1], c[2], t, Xᵢ, Vᵢ, ∇Vᵢ);
        if o > objective
            copt = c
            objective = o
        end
    end

    return copt
end

"""
Computes the optimal policy (χ', α') over the state space X
"""
function policyovergrid(t, X, V, ∇V, P)
    policyovergrid!(SharedArray{Float32, 4}((size(V)..., 2)), t, X, V, ∇V, P)
end
function policyovergrid!(policy::SVectorGrid, t, X, V, ∇V, P)
    @sync @distributed for idx ∈ CartesianIndices(V)
        Xᵢ = @view X[idx, :]
        ∇Vᵢ = @view ∇V[idx, :]
        Vᵢ = @view V[idx]

        policy[idx, :] .= optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ, P)
    end

    return policy
end
function policyovergrid!(policy::VectorGrid, t, X, V, ∇V, P)
    for idx ∈ CartesianIndices(V)
        Xᵢ = @view X[idx, :]
        ∇Vᵢ = @view ∇V[idx, :]
        Vᵢ = @view V[idx]

        policy[idx, :] .= optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ, P)
    end
    
    return policy
end

function G(t, X, V, Ω, P)
    G!(
        Array{Float32}(undef, size(V)), # ∂ₜV
        Array{Float32}(undef, size(V)..., 4), # ∇V
        Array{Float32}(undef, size(V)..., 3), # w
        SharedArray{Float32, 4}((size(V)..., 2)), # policy
        t, X, V, Ω, P
    )
end
"""
Computes G! by modifying (∂ₜV, ∇V, policy, w)
"""
function G!(∂ₜV::FieldGrid, ∇V::VectorGrid, w, policy, t, X, V, Ω, P)
    central∇!(∇V, V, Ω)
    policyovergrid!(policy, t, X, V, ∇V, P);

    χ = @view policy[:, :, :, 1];
    α = @view policy[:, :, :, 2];

    drift!(w, χ, α, t, X);
    dir∇!(∇V, V, w, Ω);

    ∂ₜV .= f.(χ, X[:, :, :, 3], V, Ref(economy)) + ∇V[:, :, :, 4] .+ ∂²(V, Ω; dim = 1) .* hogg.σ²ₜ / 2f0

    return ∂ₜV
end


