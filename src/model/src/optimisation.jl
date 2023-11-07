const optimoptions = Options(
    x_abstol = 0f0, x_reltol = 0f0, f_abstol = 0f0, f_reltol = 0f0, g_abstol = eps(Float32), g_reltol = eps(Float32), outer_x_abstol = 0f0, outer_x_reltol = 0f0, outer_f_abstol = 0f0, outer_f_reltol = 0f0, outer_g_abstol = eps(Float32), outer_g_reltol = eps(Float32),
    allow_f_increases = true, successive_f_tol = 3,
    iterations = 100, outer_iterations = 100
)

const dfc = TwiceDifferentiableConstraints([0f0, 0f0], [1f0, 1f0])

"Constructs the negative objective functional at `Xᵢ`."
function objective!(Z, ∇, H, c, t, Xᵢ, Vᵢ, ∇Vᵢ, instance::ModelInstance, calibration::Calibration)
    economy = first(instance)

    f₀, Yf₁, Y²f₂ = epsteinzinsystem(c[1], Xᵢ[3], Vᵢ[1], economy)

    εₜ = ε(t, exp(Xᵢ[2]), c[2], instance, calibration)
    Aₜ = A(t, economy)

    if !isnothing(∇)
        ∇[1] = Yf₁ + ∇Vᵢ[3] * ϕ′(t, c[1], economy)
        ∇[2] = -∇Vᵢ[2] - ∇Vᵢ[3] * Aₜ * β′(t, εₜ, economy) * ε′(t, exp(Xᵢ[2]), instance, calibration)
    
        ∇ .*= -1f0
    end

    if !isnothing(H)
        H[1, 1] = Y²f₂ + ∇Vᵢ[3] * ϕ′′(t, economy)
        H[1, 2] = 0f0
        H[2, 1] = 0f0
        H[2, 2] = -∇Vᵢ[3] * Aₜ * ε′(t, exp(Xᵢ[2]), instance, calibration)^2  * exp(-economy.ωᵣ * t)
    
        H .*= -1f0
    end

    if !isnothing(Z)
        z = f₀ + 
            ∇Vᵢ[2] * (γ(t, economy, calibration) - c[2]) + 
            ∇Vᵢ[3] * (ϕ(t, c[1], economy) - Aₜ * β(t, εₜ, economy))

        return -z
    end

        
end

"Numerically maximises the objective functional at point `Xᵢ`."
function optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ, instance::ModelInstance, calibration::Calibration; c₀ = [0.5f0, 0.5f0])

    df = TwiceDifferentiable(
        only_fgh!(
            @closure (F, ∇, H, c) -> objective!(F, ∇, H, c, t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration)
        ), c₀)
    res = optimize(df, dfc, c₀, IPNewton(), optimoptions)

    return minimizer(res)
end

"Computes the optimal `policy = (χ, α)` over the state space `X`"
function policyovergrid(t, X, V, ∇V, instance::ModelInstance, calibration::Calibration; shrink = 1f-3)

    inner = ones(Float32, size(X, 1), size(X, 2), size(X, 3), 2) ./ 2f0
    policy = BorderArray(inner, paddims(inner, 1, 1:length(size(V))))

    policyovergrid!(policy, t, X, V, ∇V, instance, calibration; shrink = shrink)
end
function policyovergrid!(policy::BorderArray, t, X, V::BorderArray, ∇V, instance::ModelInstance, calibration::Calibration; shrink = 1f-3)

    I₁ = oneunit(first(CartesianIndices(V)))
    c₀ = zeros(Float32, 2)

    for idx ∈ CartesianIndices(V.inner)
        Xᵢ = @view X[idx, :]
        ∇Vᵢ = @view ∇V[idx, :]
        Vᵢ = @view V[idx]

        c₀[1] = mean(policy[(idx - I₁):idx, 1]) * (1f0 - 2shrink) + shrink
        c₀[2] = mean(policy[(idx - I₁):idx, 2]) * (1f0 - 2shrink) + shrink
        
        policy.inner[idx, :] .= optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration; c₀ = c₀)
    end
    
    return policy
end