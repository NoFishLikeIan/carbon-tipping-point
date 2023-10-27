using Model: terminalpolicyovergrid!, ydrift!, hjbterminal
using Utils: central∂!, dir∂!, ∂²!

function G(t, X, V, Ω, instance::Model.ModelInstance, calibration::Model.Calibration)
    G!(
        similar(V), similar(V, size(V)..., 4),
        t, X, V, Ω, instance, calibration
    )
end
"Computes G! by modifying (∂ₜV, ∇V, policy, w)"
function G!(∂ₜV, tmp, t, X, V, Ω, instance::Model.ModelInstance, calibration::Model.Calibration)
    ∇V = @view tmp[:, :, 1]
    ∂²V = @view tmp[:, :, 2]
    policy = @view tmp[:, :, 3]
    w = @view tmp[:, :, 4]

    central∇!(∇V, V, Ω)
    policyovergrid!(policy, t, X, V, ∇V, instance, calibration);
    drift!(policy, α, t, X, instance, calibration);
    dir∇!(∇V, V, w, Ω);
    ∂²!(∂²V, V, Ω; dim = 1)

    economy = first(instance)

    @batch for idx in CartesianIndices(∂ₜV)
        ∂ₜV[idx] = 
            f(policy[idx, 1], X[idx, 3], V[idx], economy) +
            ∇[idx, 4] +  ∂²V[idx] * hogg.σ²ₜ / 2f0
    end

    return ∂ₜV
end

function terminalG(X, V, Ω, instance::Model.ModelInstance)
    terminalG!(
        similar(V), similar(V, size(V)..., 4),
        X, V, Ω,
        instance
    )
end
"""
Computes G! by modifying ∂ₜV and tmp = (∂yV, ∂²TV,  policy, w)
"""
function terminalG!(∂ₜV, tmp, X, V, Ω, instance::Model.ModelInstance)
    ∂yV = @view tmp[:, :, 1]
    ∂²TV = @view tmp[:, :, 2]
    policy = @view tmp[:, :, 3]
    w = @view tmp[:, :, 4]

    T = @view X[:, :, 1]

    central∂!(∂yV, V, Ω; direction = 2);
    terminalpolicyovergrid!(policy, X, V, ∂yV, instance);
    ydrift!(w, policy, T, instance);
    dir∂!(∂yV, V, w, Ω; direction = 2);
    ∂²!(∂²TV, V, Ω; dim = 1)

    for idx in CartesianIndices(∂ₜV)
        ∂ₜV[idx] = hjbterminal(policy[idx], X[idx, :], V[idx], ∂yV[idx], ∂²TV[idx], instance)
        
    end

    return ∂ₜV
end