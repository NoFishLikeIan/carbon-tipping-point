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
    twoh⁻¹ = h⁻¹ ./ 2f0

    Δ = makeΔ(dimensions(grid))
    I, R = extrema(CartesianIndices(grid))
    
   @batch for idx in CartesianIndices(grid), l in 1:dimensions(grid)
        hᵢ = ifelse(isonboundary(idx, grid, l), h⁻¹[l], twoh⁻¹[l])

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
        hᵢ = ifelse(isonboundary(idx, grid, direction), h⁻¹, twoh⁻¹)   

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
    twoh⁻¹ = h⁻¹ ./ 2f0

    d = dimensions(grid)
    sizes = size(grid)
    Δ = makeΔ(d)
    
    temp = 0f0
    @batch for idx in CartesianIndices(grid)
        temp = 0f0

        for l ∈ 1:d
            D[idx, l] = if idx.I[l] ≤ 2
                h⁻¹[l] * (V[idx + Δ[l]] - V[idx])
            elseif idx.I[l] ≥ sizes[l] - 1
                h⁻¹[l] * (V[idx] - V[idx -  Δ[l]])
            elseif w[idx, l] > 0f0
                twoh⁻¹[l] * (-3f0V[idx] + 4f0V[idx + Δ[l]] - V[idx + 2Δ[l]])
            else
                twoh⁻¹[l] * (3f0V[idx] - 4f0V[idx - Δ[l]] + V[idx - 2Δ[l]])
            end

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
function dir∂!(D, V::AbstractArray, w, grid::RegularGrid, l)
    h = steps(grid)[l]
    if any(h < ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    sizes = size(grid)
    h⁻¹ = inv(h)
    twoh⁻¹ = h⁻¹ / 2f0;

    Δᵢ = makeΔ(dimensions(grid))[l]
    
    @batch for idx in CartesianIndices(grid)

        D[idx] = if idx.I[l] ≤ 2
            h⁻¹ * (V[idx + Δᵢ] - V[idx])
        elseif idx.I[l] ≥ sizes[l] - 1
            h⁻¹ * (V[idx] - V[idx -  Δᵢ])
        elseif w[idx] > 0f0
            twoh⁻¹ * (-3f0V[idx] + 4f0V[idx + Δᵢ] - V[idx + 2Δᵢ])
        else
            twoh⁻¹ * (3f0V[idx] - 4f0V[idx - Δᵢ] + V[idx - 2Δᵢ])
        end
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
function ∂²!(D², V::AbstractArray, grid::RegularGrid, l)
    hₗ = steps(grid)[l]
    if (hₗ < ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    hₗ⁻² = inv(hₗ^2)
    sizes = size(grid)
    Δᵢ = makeΔ(dimensions(grid))[l]
    
    for idx in CartesianIndices(grid)

        D²[idx] = if idx.I[l] ≤ 1
            hₗ⁻² * (V[idx + 2Δᵢ] - 2f0V[idx + Δᵢ] + V[idx])
        elseif idx.I[l] ≥ sizes[l]
            hₗ⁻² * (V[idx] - 2f0V[idx - Δᵢ] + V[idx - 2Δᵢ])
        else
            hₗ⁻² * (V[idx + Δᵢ] - 2f0V[idx] + V[idx - Δᵢ])
        end
    end

    return D²
end