const ϵ = cbrt(eps(Float32));
"Generate basis vectors in CartesianIndex of size n"
function makeΔ(n)
    ntuple(
        i -> CartesianIndex(
            ntuple(j -> j == i ? 1 : 0, n),
        ),
        n
    )
end

"""
Given a Vₜ (n₁ × n₂ ... × nₘ) returns a matrix D (n₁ × n₂ ... × nₘ × m), with elements ∇Vₜ and last ∇Vₜ⋅w.
"""
function central∇(V, grid)
    D = Array{Float32}(undef, length.(grid)..., length(grid))
    central∇!(D, V, grid)
    return D
end
function central∇!(D, V, grid::StateGrid)
    h = steps(grid); twoh⁻¹ = inv.(2f0 .* h);
    if any(h .< ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    dimension = length(size(V))
    Δ = makeΔ(dimension)

    Iₗ, Iᵤ = CartesianIndices(V) |> extrema

    @batch for I in CartesianIndices(V), l in last(axes(D))
        D[I, l] = twoh⁻¹[l] * ( V[min(I + Δ[l], Iᵤ)] - V[max(I - Δ[l], Iₗ)] )
    end

    return D 
end

function central∂(V, grid; direction = 1)
    D = similar(V)
    central∂!(D, V, grid; direction = direction)
    return D
end
function central∂!(D, V, grid; direction = 1)
    h = steps(grid)[direction]; twoh⁻¹ = inv(2f0 .* h);
    if h < ϵ @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    Δᵢ = makeΔ(length(grid))[direction]

    Iₗ, Iᵤ = CartesianIndices(V) |> extrema

    @batch for I in CartesianIndices(V)
        D[I] = twoh⁻¹ * ( V[min(I + Δᵢ, Iᵤ)] - V[max(I - Δᵢ, Iₗ)] )
    end

    return D 
end

"""
Given a `Vₜ` `(n₁ × n₂ ... × nₘ)` and a drift `w` `(n₁ × n₂ ... × nₘ × m)` returns a matrix `D` `(n₁ × n₂ ... × nₘ × (m + 1))`, with first three elemnts ∇Vₜ and last ∇Vₜ⋅w. If `withdot = false` then ∇Vₜ⋅w is not computed and `D` has size `(n₁ × n₂ ... × nₘ × m)`

The finite difference scheme is computed by using second order forward derivatives if the drift is positive and backwards if it is negative.

"""
function dir∇(V, w, grid)
    m = length(grid) + Int(withdot ? 1 : 0)

    D = Array{Float32}(undef, length.(grid)..., m)
    dir∇!(D, V, w, grid)
    return D
end
function dir∇!(D, V, w, grid)
    h = steps(grid); twoh⁻¹ = inv.(2f0 .* h);
    if any(h .< ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    dimension = length(grid)
    Δ = makeΔ(dimension)
    
    Iₗ, Iᵤ = CartesianIndices(V) |> extrema
    temp = 0f0

    @batch for I in CartesianIndices(V)
        temp = 0f0

        for l ∈ 1:dimension
            D[I, l] = twoh⁻¹[l] * ifelse(w[I, l] > 0,
                -V[min(I + 2Δ[l], Iᵤ)] + 4f0V[min(I + Δ[l], Iᵤ)] - 3f0V[I],
                V[max(I - 2Δ[l], Iₗ)] - 4f0V[max(I - Δ[l], Iₗ)] + 3f0V[I]
            )

            temp += D[I, l] * w[I, l]
        end

        D[I, dimension + 1] = temp
    end

    return D 
end

function dir∂(V, w, grid; direction = 1)
    D = similar(V)
    dir∂!(D, V, w, grid; direction = direction)
    return D
end
function dir∂!(D, V, w, grid; direction = 1)
    h = steps(grid)[direction]; twoh⁻¹ = inv(2f0 .* h);
    if h < ϵ @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    Δᵢ = makeΔ(length(grid))[direction]

    Iₗ, Iᵤ = CartesianIndices(V) |> extrema

    @batch for I in CartesianIndices(V)
        D[I, 1] = twoh⁻¹ * ifelse(w[I] > 0,
            -V[min(I + 2Δᵢ, Iᵤ)] + 4f0V[min(I + Δᵢ, Iᵤ)] - 3f0V[I],
            V[max(I - 2Δᵢ, Iₗ)] - 4f0V[max(I - Δᵢ, Iₗ)] + 3f0V[I]
        )
    end

    return D 
end

"""
Given a Vₜ (n₁ × n₂ × n₃) computes the second derivative in the direction of the l-th input xₗ.
"""
function ∂²(V, grid; dim = 1)
    D² = similar(V)
    ∂²!(D², V, grid; dim = dim)
    return D²
end
function ∂²!(D², V, grid; dim = 1)
    hₗ = steps(grid)[dim]; hₗ⁻² = inv(hₗ^2)
    if (hₗ < ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    dimensions = length(grid)
    Δᵢ = makeΔ(dimensions)[dim]

    Iₗ, Iᵤ = CartesianIndices(V) |> extrema
    
    @batch for idx in CartesianIndices(V)
        D²[idx] = hₗ⁻² * (
            V[min(idx + Δᵢ, Iᵤ)] - 2f0 * V[idx] + V[max(idx - Δᵢ, Iₗ)]
        )
    end

    return D² 
end