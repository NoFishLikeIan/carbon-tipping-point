function G(t, X, V, Ω, instance::Model.ModelInstance, calibration::Model.Calibration)
    G!(
        similar(V), similar(V, size(V)..., 4),
        t, X, V, Ω, instance, calibration
    )
end
"Computes G! by modifying (∂ₜV, ∇V, policy, w)"
function G!(∂ₜV, tmp, t, X, V, Ω, instance::Model.ModelInstance, calibration::Model.Calibration)
    economy = first(instance)

    central∇!(∇V, V, Ω)
    policyovergrid!(policy, t, X, V, ∇V, instance, calibration);
    drift!(policy, α, t, X, instance, calibration);
    dir∇!(∇V, V, w, Ω);

    ∂ₜV .= f.(χ, X[:, :, :, 3], V, Ref(economy)) + ∇V[:, :, :, 4] .+ ∂²(V, Ω; dim = 1) .* hogg.σ²ₜ / 2f0

    return ∂ₜV
end

function terminalG(X, V, Ω, instance::Model.ModelInstance, calibration::Model.Calibration)
    terminalG!(
        similar(V), similar(V, size(V)..., 4),
        X, V, Ω,
        instance, calibration
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
    y = @view X[:, :, 2]

    Utils.central∂!(∂yV, V, Ω; direction = 2);
    Model.terminalpolicyovergrid!(policy, X, V, ∂yV, instance);
    Model.ydrift!(w, policy, T, instance);
    Utils.dir∂!(∂yV, V, w, Ω; direction = 2);
    Utils.∂²!(∂²TV, V, Ω; dim = 1)


    for idx in CartesianIndices(∂ₜV)
        ∂ₜV[idx] = Model.hjbterminal(policy[idx], X[idx, :], V[idx], ∂yV[idx], ∂²TV[idx], instance)
    end

    return ∂ₜV
end