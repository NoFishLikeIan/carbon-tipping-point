using Utils: ∂²!, central∇!, dir∇!, central∂!, dir∂!
using Model: terminalpolicyovergrid!, ȳdrift!, hjbterminal
using Model: policyovergrid!, drift!
using Model: ModelInstance, Calibration
using Polyester: @batch

"Computes G! by modifying (∂ₜV, ∇V, policy, w)"
function G!(∂ₜV, ∇V, ∂²V, policy, w, t, X, V, Ω, instance::ModelInstance, calibration::Calibration)
    central∇!(∇V, V, Ω)
    policyovergrid!(policy, t, X, V, ∇V, instance, calibration);
    drift!(w, t, X, policy, instance, calibration);
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

function terminalG(X, V::BorderArray, Ω, instance::ModelInstance)
    ∂ₜV = similar(V.inner)
    ∂yV = similar(V.inner)
    ∂²yV = similar(V.inner)
    ẏ = similar(V.inner)
    χ = similar(V.inner)

    terminalG!(
        ∂ₜV, ∂yV, ∂²yV, ẏ, χ,
        X, V, Ω,
        instance
    )
end
"""
Computes G! by modifying ∂ₜV. Takes as input X, V, Ω. Also modifies the derivatives matrices (∂V∂y, ∂V∂T, ∂²V∂T²) the policy χ and the drift ẏ.
"""
function terminalG!(
    ∂ₜV, V::BorderArray, 
    ∂V∂T, ∂V∂y, ∂²V∂T², χ, ẏ,
    grid::RegularGrid, instance::ModelInstance)
    
    economy = first(instance)

    central∂!(∂V∂y, V, grid, 3)
    terminalpolicyovergrid!(χ, V, ∂V∂y, grid, economy)
    ȳdrift!(ẏ, χ, grid, instance)
    
    dir∂!(∂V∂T, V, ẏ, grid, 1)
    dir∂!(∂V∂y, V, ẏ, grid, 3)
    ∂²!(∂²V∂T², V, grid, 1);

    @batch for I in CartesianIndices(grid)
        Xᵢ = @view grid.X[I, :]
        ∂ₜV[I] = -hjbterminal(χ[I], Xᵢ, V[I], ∂V∂T[I], ∂V∂y[I], ∂²V∂T²[I], instance)
    end

    return ∂ₜV
end