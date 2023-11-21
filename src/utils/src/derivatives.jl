const ϵ = cbrt(eps(Float32));

"Generate basis vectors in CartesianIndex of size n"
function makeΔ(n)
    ntuple(i -> CartesianIndex(ntuple(j -> j == i ? 1 : 0, n)), n)
end

"""
Given a Vₜ (n₁ × n₂ ... × nₘ) returns a matrix D (n₁ × n₂ ... × nₘ × m), with elements ∇Vₜ and last ∇Vₜ⋅w.
"""
function central∇(V::AbstractArray, grid::RegularGrid)
    D = Array{Float32}(undef, size(grid)..., dimensions(grid) + 1)
    central∇!(D, V, grid)
end
function central∇!(D, V::AbstractArray, grid::RegularGrid)
    h = steps(grid)
    if any(h .< ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    h⁻¹ = inv.(h)
    twoh⁻¹ = h⁻¹ ./ 2f0;

    Δ = makeΔ(dimensions(grid))
    I, R = extrema(CartesianIndices(grid))
    
   @batch for idx in CartesianIndices(grid), l in 1:dimensions(grid)
        hᵢ = ifelse(isonboundary(idx, grid), h⁻¹[l], twoh⁻¹[l])   

        D[idx, l] = hᵢ * ( 
            V[min(idx + Δ[l], R)] - 
            V[max(idx - Δ[l], I)] 
        )
    end

    return D
end

function central∂(V::AbstractArray, grid::RegularGrid, direction)
    D = Array{Float32}(undef, size(grid))
    central∂!(D, V, grid, direction)
end
function central∂!(D, V::AbstractArray, grid::RegularGrid, direction)
    h = steps(grid)[direction]
    if h < ϵ @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    h⁻¹ = inv(h)
    twoh⁻¹ = h⁻¹ / 2f0

    Δᵢ = makeΔ(dimensions(grid))[direction]
    I, R = extrema(CartesianIndices(grid))

    @batch for idx in CartesianIndices(grid)
        hᵢ = ifelse(isonboundary(idx, grid), h⁻¹, twoh⁻¹)   

        D[idx] = hᵢ * ( 
            V[min(idx + Δᵢ, R)] - 
            V[max(idx - Δᵢ, I)] 
        )
    end

    return D 
end

"""
Given a `Vₜ` `(n₁ × n₂ ... × nₘ)` and a drift `w` `(n₁ × n₂ ... × nₘ × m)` returns a matrix `D` `(n₁ × n₂ ... × nₘ × (m + 1))`, with first three elements ∇Vₜ and last ∇Vₜ⋅w.

The finite difference scheme is computed by using second order forward derivatives if the drift is positive and backwards if it is negative.
"""
function dir∇(V::AbstractArray, w, grid::RegularGrid)
    D = Array{Float32}(undef, size(grid)..., dimensions(grid) + 1)
    dir∇!(D, V, w, grid)
end
function dir∇!(D, V::AbstractArray, w, grid::RegularGrid)
    h = steps(grid)
    if any(h .< ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    h⁻¹ = inv.(h)
    twoh⁻¹ = h⁻¹ ./ 2f0;

    d = dimensions(grid)
    Δ = makeΔ(d)
    I, R = extrema(CartesianIndices(grid))
    
    temp = 0f0
    @batch for idx in CartesianIndices(grid)
        temp = 0f0

        for l ∈ 1:d
            hᵢ = ifelse(isonboundary(idx, grid), h⁻¹[l], twoh⁻¹[l])

            D[idx, l] = hᵢ * ifelse(
                w[idx, l] > 0f0,
                -V[min(idx + 2Δ[l], R)] + 4f0V[min(idx + Δ[l], R)] - 3f0V[idx],
                V[max(idx - 2Δ[l], I)] - 4f0V[max(idx - Δ[l], I)] + 3f0V[idx]
            )

            temp += D[idx, l] * w[idx, l]
        end

        D[idx, d + 1] = temp
    end

    return D 
end

function dir∂(V::AbstractArray, w, grid::RegularGrid, direction)
    D = Array{Float32}(undef, size(grid))
    dir∂!(D, V, w, grid, direction)
end
function dir∂!(D, V::AbstractArray, w, grid::RegularGrid, direction)
    h = steps(grid)[direction]
    if any(h < ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    h⁻¹ = inv(h)
    twoh⁻¹ = h⁻¹ / 2f0;

    Δᵢ = makeΔ(dimensions(grid))[direction]
    I, R = extrema(CartesianIndices(grid))
    
    @batch for idx in CartesianIndices(grid)
        hᵢ = ifelse(isonboundary(idx, grid), h⁻¹, twoh⁻¹)

        D[idx] = hᵢ * ifelse(
            w[idx] > 0f0,
            -V[min(idx + 2Δᵢ, R)] + 4f0V[min(idx + Δᵢ, R)] - 3f0V[idx],
            V[max(idx - 2Δᵢ, I)] - 4f0V[max(idx - Δᵢ, I)] + 3f0V[idx]
        )
    end

    return D 
end

"""
Given a Vₜ (n₁ × n₂ × n₃) computes the second derivative in the direction of the l-th input xₗ.
"""
function ∂²(V::AbstractArray, grid::RegularGrid, direction)
    D² = Array{Float32}(undef, size(grid))
    ∂²!(D², V, grid, direction)
    return D²
end
function ∂²!(D², V::AbstractArray, grid::RegularGrid, direction)
    hₗ = steps(grid)[direction]
    if (hₗ < ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    hₗ⁻² = inv(hₗ^2)
    Δᵢ = makeΔ(dimensions(grid))[direction]
    I, R = extrema(CartesianIndices(grid))
    
    @batch for idx in CartesianIndices(grid)
        hᵢ = ifelse(isonboundary(idx, grid), 2hₗ⁻², hₗ⁻²)
        D²[idx] = hᵢ * (
            V[min(idx + Δᵢ, R)] - 2f0 * V[idx] + V[max(idx - Δᵢ, I)]
        )
    end

    return D² 
end