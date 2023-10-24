using Polyester: @batch
using Optim: optimize, minimizer, Newton

"""
Computes the Hamilton-Jacobi-Bellmann equation at point Xᵢ
"""
function hjb(c, t, Xᵢ, Vᵢ, ∇Vᵢ, ∂²Vᵢ)
    f(c[1], Xᵢ[3], Vᵢ[1], economy) + 
        ∇Vᵢ[1] * μ(Xᵢ[1], Xᵢ[2], hogg, albedo) +
        ∇Vᵢ[2] * (γᵇ(t) - c[2]) + 
        ∇Vᵢ[3] * (
            economy.ϱ + ϕ(t, c[1], economy) - 
            A(t, economy) * ε(t, exp(Xᵢ[2]), c[2])^2 * exp(-economy.ωᵣ * t) / 2 -
            δₖ(Xᵢ[1], economy, hogg)
        ) +
        ∂²Vᵢ[1] * hogg.σ²ₜ / 2f0
end

"""
Computes the negative objective functional at point Xᵢ which depends on the control.
"""
function objective(t, Xᵢ, Vᵢ, ∇Vᵢ)
    c -> -f(c[1], Xᵢ[3], Vᵢ[1], economy) - 
        ∇Vᵢ[2] * (γᵇ(t) - c[2]) - 
        ∇Vᵢ[3] * (
            ϕ(t, c[1], economy) - 
            A(t, economy) * ε(t, exp(Xᵢ[2]), c[2])^2 * exp(-economy.ωᵣ * t) / 2
        )
end


function gradientobjective(t, Xᵢ, Vᵢ, ∇Vᵢ)
    function ∇J!(∇, c)
        ∇[1] = -Y∂f(c[1], Xᵢ[3], Vᵢ[1], economy) - ∇Vᵢ[3] * ϕ′(t, c[1], economy)
        ∇[2] = ∇Vᵢ[2] + A(t, economy) * ε(t, exp(Xᵢ[2]), c[2]) * ε′(t, exp(Xᵢ[2])) * exp(-economy.ωᵣ * t)
    end
end
function hessianobjective(t, Xᵢ, Vᵢ, ∇Vᵢ)
    function ∇H!(H, c)
        H[1, 1] = - Y²∂²f(c[1], Xᵢ[3], Vᵢ[1], economy) - ∇Vᵢ[3] * ϕ′′(t, economy)
        H[1, 2] = 0f0
        H[2, 1] = 0f0
        H[2, 2] = ∇Vᵢ[2] + ε′(t, exp(Xᵢ[2]))^2
    end
end

"""
Computes the the objective functional at point Xᵢ which depends on the control.
"""
function objectivefunction(c, t, Xᵢ, Vᵢ, ∇Vᵢ)
    f(c[1], Xᵢ[3], Vᵢ[1], economy) + 
        ∇Vᵢ[2] * (γᵇ(t) - c[2]) + 
        ∇Vᵢ[3] * (
            ϕ(c[1],  A(t, economy), economy) -  A(t, economy) * β(t, ε(t, exp(Xᵢ[2]), c[2]), economy)
        )
end

function optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ; c₀ = [0.5f0, 0.5f0])
    df = TwiceDifferentiable(
        objective(t, Xᵢ, Vᵢ, ∇Vᵢ),
        gradientobjective(t, Xᵢ, Vᵢ, ∇Vᵢ),
        hessianobjective(t, Xᵢ, Vᵢ, ∇Vᵢ),
        c₀
    )

    dfc = TwiceDifferentiableConstraints(
        [0f0, 0f0], [1f0, 1f0]
    )

    res = optimize(
        df, dfc, c₀, IPNewton(),
        Optim.Options(
            x_abstol = 0f0,
            x_reltol = 0f0,
            f_abstol = 0f0,
            f_reltol = 0f0,
            g_abstol = 1f-8,
            g_reltol = 1f-8,
            outer_x_abstol = 0f0,
            outer_x_reltol = 0f0,
            outer_f_abstol = 0f0,
            outer_f_reltol = 0f0,
            outer_g_abstol = 1f-8,
            outer_g_reltol = 1f-8,
        )
    )

    return minimizer(res)
end

"""
Computes the the optimal policy at point Xᵢ by computing all points in P. P is assumed to be a Latin Hyper Grid.
"""
function optimalpolicygreedy(t, Xᵢ, Vᵢ, ∇Vᵢ, P)
    z = -Inf32
    copt = copy(P[1, :])

    for i ∈ axes(P, 1)
        c = @view P[i, :]
        zᵢ = objectivefunction(c, t, Xᵢ, Vᵢ, ∇Vᵢ);
        if zᵢ > z
            copt .= c
            z = zᵢ
        end
    end

    return copt
end

"""
Computes the optimal policy (χ', α') over the state space X
"""
function policyovergrid(t, X, V, ∇V, P)
    policyovergrid!(Array{Float32, 4}(undef, size(V)..., 2), t, X, V, ∇V, P)
end
function policyovergrid!(policy::VectorGrid, t, X, V, ∇V, P)
    @batch for idx ∈ CartesianIndices(V)
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


