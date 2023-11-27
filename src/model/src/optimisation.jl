const optimoptions = Options(
    x_abstol = 0f0, x_reltol = 0f0, f_abstol = 0f0, f_reltol = 0f0, g_abstol = eps(Float32), g_reltol = eps(Float32), outer_x_abstol = 0f0, outer_x_reltol = 0f0, outer_f_abstol = 0f0, outer_f_reltol = 0f0, outer_g_abstol = eps(Float32), outer_g_reltol = eps(Float32),
    allow_f_increases = true, successive_f_tol = 3,
    iterations = 100, outer_iterations = 100
)

const dfc = TwiceDifferentiableConstraints([0f0, 0f0], [1f0, 1f0])

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
function policyovergrid(t, X, V, ∇V, instance::ModelInstance, calibration::Calibration)
    policy = ones(Float32, size(X, 1), size(X, 2), size(X, 3), 2) ./ 2f0
    policyovergrid!(policy, t, X, V, ∇V, instance, calibration)
end
function policyovergrid!(policy, t, X, V, ∇V, instance::ModelInstance, calibration::Calibration)

    c₀ = zeros(Float32, 2)

    for idx ∈ CartesianIndices(V.inner)
        Xᵢ = @view X[idx, :]
        ∇Vᵢ = @view ∇V[idx, :]
        Vᵢ = @view V[idx]
        
        policy.inner[idx, :] .= optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration; c₀ = c₀)
    end
    
    return policy
end

function optimalterminalpolicy(Xᵢ, Vᵢ::Real, ∂V∂yᵢ::Real, economy::Economy; tol = 1f-3)
    g = @closure χ -> terminalfoc(χ, Xᵢ, Vᵢ, ∂V∂yᵢ, economy) 
    first(bisection(g, tol, 1f0 - tol))
end

function terminalpolicyovergrid!(policy, V, ∂yV::AbstractArray, grid::RegularGrid, economy::Economy)
    @batch for idx ∈ CartesianIndices(grid)
        Xᵢ = @view grid.X[idx, :]
        policy[idx] = optimalterminalpolicy(Xᵢ, V[idx], ∂yV[idx], economy)
    end
end