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



function terminalG(X, V, Ω)
    terminalG!(
        similar(V), similar(V, size(V)..., 4),
        X, V, Ω
    )
end
"""
Computes G! by modifying ∂ₜV and tmp = (∂yV, ∂²TV,  policy, w)
"""
function terminalG!(∂ₜV, tmp, X, V, Ω)
    ∂yV = @view tmp[:, :, 1]
    ∂²TV = @view tmp[:, :, 2]
    policy = @view tmp[:, :, 3]
    w = @view tmp[:, :, 4]
    T = @view X[:, :, 1]
    y = @view X[:, :, 2]

    central∂!(∂yV, V, Ω; direction = 2);
    terminalpolicyovergrid!(policy, X, V, ∂yV);
    ydrift!(w, policy, T)
    dir∂!(∂yV, V, w, Ω; direction = 2);
    ∂²!(∂²TV, V, Ω; dim = 1)


    for idx in CartesianIndices(∂ₜV)
        ∂ₜV[idx] = hjbterminal(policy[idx], X[idx, :], V[idx], ∂yV[idx], ∂²TV[idx])
    end

    return ∂ₜV
end