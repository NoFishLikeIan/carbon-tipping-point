"Emissivity rate implied by abatement `α` at time `t` and carbon concentration `M`"
function ε(t, M, α, instance::ModelInstance, calibration::Calibration)
    economy, hogg, _ = instance

    1f0 - M * (δₘ(M, hogg) + γ(t, economy, calibration) - α) / (Gtonoverppm * Eᵇ(t, economy, calibration))
end

function ε′(t, M, instance::ModelInstance, calibration::Calibration)
    M / (Gtonoverppm * Eᵇ(t, instance[1], calibration))
end

"Computes drift over whole state space `X`"
drift(t, X, policy::AbstractArray, instance::ModelInstance, calibration::Calibration) = drift!(similar(X), t, X, policy, instance, calibration)
function drift!(w, t, X, policy::AbstractArray, instance::ModelInstance, calibration::Calibration)
    economy, hogg, albedo = instance

    γₜ = γ(t, economy, calibration)
    Aₜ = A(t, economy)

    dimensions = size(X)[1:3]

    @batch for idx in CartesianIndices(dimensions)
        c = @view policy[idx, :]

        w[idx, 1] = μ(X[idx, 1], X[idx, 2], hogg, albedo)
        w[idx, 2] = γₜ - c[2]
        w[idx, 3] = economy.ϱ + ϕ(c[1], Aₜ, economy) -
            Aₜ *  β(t, ε(t, exp(X[idx, 2]), c[2], instance, calibration), economy) - 
            δₖ(X[idx, 1], economy, hogg)

    end

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
            A(t, economy) * β(t, ε(t, exp(Xᵢ[2]), c[2], instance, calibration), economy) -
            δₖ(Xᵢ[1], economy, hogg)
        ) +
        ∂²Vᵢ[1] * hogg.σ²ₜ / 2f0
end

"Constructs the negative objective functional at `Xᵢ`."
function objective(c, t, Xᵢ, Vᵢ, ∇Vᵢ, instance::ModelInstance, calibration::Calibration)
    economy = first(instance)

    z = f(c[1], Xᵢ[3], Vᵢ[1], economy) + 
        ∇Vᵢ[2] * (γ(t, economy, calibration) - c[2]) + 
        ∇Vᵢ[3] * (ϕ(t, c[1], economy) - A(t, economy) * β(t, ε(t, exp(Xᵢ[2]), c[2], instance, calibration), economy))

    return -z
end

"Constructs the gradient function of the objective functional at `Xᵢ`."
function gradientobjective!(∇, c, t, Xᵢ, Vᵢ, ∇Vᵢ, instance::ModelInstance, calibration::Calibration)
    economy = first(instance)

    ∇[1] = Y∂f(c[1], Xᵢ[3], Vᵢ[1], economy) + ∇Vᵢ[3] * ϕ′(t, c[1], economy)
    ∇[2] = -∇Vᵢ[2] - ∇Vᵢ[3] * A(t, economy) * β′(t, ε(t, exp(Xᵢ[2]), c[2], instance, calibration), economy) * ε′(t, exp(Xᵢ[2]), instance, calibration)

    ∇ .*= -1f0
end

"Constructs the hessian function of the objective functional at `Xᵢ`." # TODO: Check calculations
function hessianobjective!(H, c, t, Xᵢ, Vᵢ, ∇Vᵢ, instance::ModelInstance, calibration::Calibration)
    economy = first(instance)

    H[1, 1] = Y²∂²f(c[1], Xᵢ[3], Vᵢ[1], economy) + ∇Vᵢ[3] * ϕ′′(t, economy)
    H[1, 2] = 0f0
    H[2, 1] = 0f0
    H[2, 2] = -∇Vᵢ[3] * A(t, economy) * ε′(t, exp(Xᵢ[2]), instance, calibration)^2  * exp(-economy.ωᵣ * t)

    H .*= -1f0
end

const optimoptions = Options(
    x_abstol = 0f0, x_reltol = 0f0, f_abstol = 0f0, f_reltol = 0f0, g_abstol = 1f-8, g_reltol = 1f-8, outer_x_abstol = 0f0, outer_x_reltol = 0f0, outer_f_abstol = 0f0, outer_f_reltol = 0f0, outer_g_abstol = 1f-8, outer_g_reltol = 1f-8,
    allow_f_increases = true, successive_f_tol = 3
)

const dfc = TwiceDifferentiableConstraints([1f-3, 1f-3], [1f0 - 1f-3, 1f0 - 1f-3])

"Numerically maximises the objective functional at point `Xᵢ`."
function optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ, instance::ModelInstance, calibration::Calibration; c₀ = [0.5f0, 0.5f0])
    g = @closure c -> objective(c, t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration)
    ∇g! = @closure (∇, c) -> gradientobjective!(∇, c, t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration)
    Hg! = @closure (H, c) ->  hessianobjective!(H, c, t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration)

    df = TwiceDifferentiable(g, ∇g!, Hg!, c₀)
    res = optimize(df, dfc, c₀, IPNewton(), optimoptions)

    return minimizer(res)
end

"Computes the optimal `policy = (χ, α)` over the state space `X`"
function policyovergrid(t, X, V, ∇V, instance::ModelInstance, calibration::Calibration)

    inner = ones(Float32, size(X, 1), size(X, 2), size(X, 3), 2) ./ 2f0
    policy = BorderArray(inner, paddims(inner, 1, 1:length(size(V))))

    policyovergrid!(policy, t, X, V, ∇V, instance, calibration)
end
function policyovergrid!(policy::BorderArray, t, X, V::BorderArray, ∇V, instance::ModelInstance, calibration::Calibration)

    I₁ = oneunit(first(CartesianIndices(V)))

    for idx ∈ CartesianIndices(V.inner)
        Xᵢ = @view X[idx, :]
        ∇Vᵢ = @view ∇V[idx, :]
        Vᵢ = @view V[idx]

        χ₀, α₀ = policy[idx - I₁, :]

        policy.inner[idx, :] .= optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration; c₀ = [χ₀, α₀])
    end
    
    return policy
end
