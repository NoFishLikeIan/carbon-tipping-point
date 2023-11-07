const ϵ = cbrt(eps(Float32));

"Generate basis vectors in CartesianIndex of size n"
function makeΔ(n)
    ntuple(i -> CartesianIndex(ntuple(j -> j == i ? 1 : 0, n)), n)
end

"Constructs a `Pad` object of the dimension of A for the first three coordinates"
function paddims(A::AbstractArray, padding::Int, dims = 1:length(size(A)))
    Pad(ntuple(i -> i ∈ dims ? padding : 0, length(size(A))))
end

"""
Given a Vₜ (n₁ × n₂ ... × nₘ) returns a matrix D (n₁ × n₂ ... × nₘ × m), with elements ∇Vₜ and last ∇Vₜ⋅w.
"""
function central∇(V::AbstractArray, grid)
    D = Array{Float32}(undef, length.(grid)..., length(grid))
    central∇!(D, V, grid)
end
function central∇!(D, V::AbstractArray, grid)
    central∇!(D, BorderArray(V, paddims(V, 1)), grid)
end
function central∇!(D, V::BorderArray, grid)
    h = steps(grid); twoh⁻¹ = inv.(2f0 .* h);
    if any(h .< ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    dimensions = length(grid)
    Δ = makeΔ(dimensions)

    @batch for I in CartesianIndices(V.inner), l in 1:dimensions
        D[I, l] = twoh⁻¹[l] * ( V[I + Δ[l]] - V[I - Δ[l]] )
    end

    return D
end

function central∂(V::AbstractArray, grid; direction = 1)
    D = Array{Float32}(undef, length.(grid))
    central∂!(D, V, grid; direction = direction)
end
function central∂!(D, V::AbstractArray, grid; direction = 1)
    central∂!(D, BorderArray(V, paddims(V, 1)), grid; direction = direction)
end
function central∂!(D, V::BorderArray, grid; direction = 1)
    h = steps(grid)[direction]; twoh⁻¹ = inv(2f0 .* h);
    if h < ϵ @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    dimensions = length(grid)
    Δᵢ = makeΔ(dimensions)[direction]

    for I in CartesianIndices(V.inner)
        D[I] = twoh⁻¹ * (V[I + Δᵢ] - V[I - Δᵢ])
    end

    return D 
end

"""
Given a `Vₜ` `(n₁ × n₂ ... × nₘ)` and a drift `w` `(n₁ × n₂ ... × nₘ × m)` returns a matrix `D` `(n₁ × n₂ ... × nₘ × (m + 1))`, with first three elements ∇Vₜ and last ∇Vₜ⋅w. If `withdot = false` then ∇Vₜ⋅w is not computed and `D` has size `(n₁ × n₂ ... × nₘ × m)`

The finite difference scheme is computed by using second order forward derivatives if the drift is positive and backwards if it is negative.
"""
function dir∇(V::AbstractArray, w, grid)
    D = Array{Float32}(undef, length.(grid)..., m)
    dir∇!(D, V, w, grid)
end
function dir∇!(D, V::AbstractArray, w, grid)
    dir∇!(D, BorderArray(V, paddims(V, 2)), w, grid)
end
function dir∇!(D, V::BorderArray, w, grid)
    h = steps(grid); twoh⁻¹ = inv.(2f0 .* h);
    if any(h .< ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    dimension = length(grid)
    Δ = makeΔ(dimension)
    
    temp = 0f0
    @batch for I in CartesianIndices(V.inner)
        temp = 0f0

        for l ∈ 1:dimension
            D[I, l] = twoh⁻¹[l] * ifelse(
                w[I, l] > 0,
                -V[I + 2Δ[l]] + 4f0V[I + Δ[l]] - 3f0V[I],
                V[I - 2Δ[l]] - 4f0V[I - Δ[l]] + 3f0V[I]
            )

            temp += D[I, l] * w[I, l]
        end

        D[I, dimension + 1] = temp
    end

    return D 
end

function dir∂(V::AbstractArray, w, grid; direction = 1)
    D = similar(V)
    dir∂!(D, V, w, grid; direction = direction)
end
function dir∂!(D, V::AbstractArray, w, grid; direction = 1)
    dir∂!(D, BorderArray(V, paddims(V, 2)), w, grid; direction = direction)
end
function dir∂!(D, V::BorderArray, w, grid; direction = 1)
    h = steps(grid)[direction]; twoh⁻¹ = inv(2f0 .* h);
    if h < ϵ @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    Δᵢ = makeΔ(length(grid))[direction]

    @batch for I in CartesianIndices(V.inner)
        D[I, 1] = twoh⁻¹ * ifelse(
            w[I] > 0,
            -V[I + 2Δᵢ] + 4f0V[I + Δᵢ] - 3f0V[I],
            V[I - 2Δᵢ] - 4f0V[I - Δᵢ] + 3f0V[I]
        )
    end

    return D 
end

"""
Given a Vₜ (n₁ × n₂ × n₃) computes the second derivative in the direction of the l-th input xₗ.
"""
function ∂²(V::AbstractArray, grid; dim = 1)
    D² = similar(V)
    ∂²!(D², V, grid; dim = dim)
    return D²
end
function ∂²!(D, V::AbstractArray, grid; dim = 1)
    ∂²!(D, BorderArray(V, paddims(V, 2)), grid; dim = dim)
end
function ∂²!(D², V::BorderArray, grid; dim = 1)
    hₗ = steps(grid)[dim]; hₗ⁻² = inv(hₗ^2)
    if (hₗ < ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    dimensions = length(grid)
    Δᵢ = makeΔ(dimensions)[dim]
    
    @batch for I in CartesianIndices(V.inner)
        D²[I] = hₗ⁻² * (V[I + Δᵢ] - 2f0 * V[I] + V[I - Δᵢ])
    end

    return D² 
end