"Computes drift over whole state space `X`"
drift(policy, t, X, instance::ModelInstance, calibration::Calibration) = drift!(similar(X), policy, t, X, instance, calibration)
function drift!(w, policy, t, X, instance::ModelInstance, calibration::Calibration)
    economy, hogg, albedo = instance

    T = @view X[:, :, :, 1]
    m = @view X[:, :, :, 2]
    χ = @view policy[:, :, :, 1]
    α = @view policy[:, :, :, 2]

    w[:, :, :, 1] .= μ.(T, m, Ref(hogg), Ref(albedo))
    w[:, :, :, 2] .= γ(t, economy, calibration) .- α
    w[:, :, :, 3] .= economy.ϱ .+ ϕ.(χ, A(t, economy), Ref(economy)) .- 
        A(t, economy) .* β.(t, ε(t, exp.(m), α, instance, albedo), Ref(economy)) .- 
        δₖ.(T, Ref(economy), Ref(hogg))

    return w
end

"""
Computes the Hamilton-Jacobi-Bellmann equation at point `Xᵢ`.
"""
function hjb(c, t, Xᵢ, Vᵢ, ∇Vᵢ, ∂²Vᵢ, instance::ModelInstance, calibration::Calibration)
    economy, hogg, albedo = instance

    f(c[1], Xᵢ[3], Vᵢ[1], economy) + 
        ∇Vᵢ[1] * μ(Xᵢ[1], Xᵢ[2], hogg, albedo) +
        ∇Vᵢ[2] * (γ(t, economy, calibration) - c[2]) + 
        ∇Vᵢ[3] * (
            economy.ϱ + ϕ(t, c[1], economy) - 
            A(t, economy) * ε(t, exp(Xᵢ[2]), c[2], instance, calibration)^2 * exp(-economy.ωᵣ * t) / 2 -
            δₖ(Xᵢ[1], economy, hogg)
        ) +
        ∂²Vᵢ[1] * hogg.σ²ₜ / 2f0
end

"Constructs the negative objective functional at `Xᵢ`."
function objective(t, Xᵢ, Vᵢ, ∇Vᵢ, instance::ModelInstance, calibration::Calibration)
    economy = first(instance)
    function J(c)
        -f(c[1], Xᵢ[3], Vᵢ[1], economy) - 
            ∇Vᵢ[2] * (γ(t, economy, calibration) - c[2]) + 
            ∇Vᵢ[3] * (
                ϕ(t, c[1], economy) - 
                A(t, economy) * ε(t, exp(Xᵢ[2]), c[2], instance, calibration)^2 * exp(-economy.ωᵣ * t) / 2
            )
    end
end

"Constructs the gradient function of the objective functional at `Xᵢ`."
function gradientobjective(t, Xᵢ, Vᵢ, ∇Vᵢ, instance::ModelInstance, calibration::Calibration)
    economy = first(instance)
    function ∇J!(∇, c)
        ∇[1] = -Y∂f(c[1], Xᵢ[3], Vᵢ[1], economy) - ∇Vᵢ[3] * ϕ′(t, c[1], economy)
        ∇[2] = ∇Vᵢ[2] + A(t, economy) * ε(t, exp(Xᵢ[2]), c[2], instance, calibration) * ε′(t, exp(Xᵢ[2]), instance, calibration) * exp(-economy.ωᵣ * t)
    end
end

"Constructs the hessian function of the objective functional at `Xᵢ`." # TODO: Check calculations
function hessianobjective(t, Xᵢ, Vᵢ, ∇Vᵢ, instance::ModelInstance, calibration::Calibration)
    economy = first(instance)
    function ∇H!(H, c)
        H[1, 1] = - Y²∂²f(c[1], Xᵢ[3], Vᵢ[1], economy) - ∇Vᵢ[3] * ϕ′′(t, economy)
        H[1, 2] = 0f0
        H[2, 1] = 0f0
        H[2, 2] = A(t, economy) * ε′(t, exp(Xᵢ[2]), instance, calibration)^2  * exp(-economy.ωᵣ * t)
    end
end

"Numerically maximises the objective functional at point `Xᵢ`."
function optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ, instance::ModelInstance, calibration::Calibration; c₀ = [0.5f0, 0.5f0])
    df = TwiceDifferentiable(
        objective(t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration),
        gradientobjective(t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration),
        hessianobjective(t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration),
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

"Computes the optimal `policy = (χ, α)` over the state space `X`"
function policyovergrid(t, X, V, ∇V, instance::ModelInstance, calibration::Calibration)
    policyovergrid!(Array{Float32, 4}(undef, size(V)..., 2), t, X, V, ∇V, instance, calibration)
end
function policyovergrid!(policy, t, X, V, ∇V, instance::ModelInstance, calibration::Calibration)
    @batch for idx ∈ CartesianIndices(V)
        Xᵢ = @view X[idx, :]
        ∇Vᵢ = @view ∇V[idx, :]
        Vᵢ = @view V[idx]

        policy[idx, :] .= optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration) # TODO: Set guess c₀
    end
    
    return policy
end

function G(t, X, V, Ω, instance::ModelInstance, calibration::Calibration)
    G!(
        Array{Float32}(undef, size(V)), # ∂ₜV
        Array{Float32}(undef, size(V)..., 4), # ∇V
        Array{Float32}(undef, size(V)..., 3), # w
        SharedArray{Float32, 4}((size(V)..., 2)), # policy
        t, X, V, Ω, instance, calibration
    )
end
"Computes G! by modifying (∂ₜV, ∇V, policy, w)"
function G!(∂ₜV, ∇V, w, policy, t, X, V, Ω, instance::ModelInstance, calibration::Calibration)
    economy = first(instance)

    central∇!(∇V, V, Ω)
    policyovergrid!(policy, t, X, V, ∇V, instance, calibration);
    drift!(policy, α, t, X, instance, calibration);
    dir∇!(∇V, V, w, Ω);

    ∂ₜV .= f.(χ, X[:, :, :, 3], V, Ref(economy)) + ∇V[:, :, :, 4] .+ ∂²(V, Ω; dim = 1) .* hogg.σ²ₜ / 2f0

    return ∂ₜV
end