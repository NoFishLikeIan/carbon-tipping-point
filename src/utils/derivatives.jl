using Polyester: @batch

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
Given a Vₜ (n₁ × n₂ × n₃) returns a matrix D (n₁ × n₂ × n₃ × 3), with elements ∇Vₜ and last ∇Vₜ⋅w.
"""
function central∇(V, grid)
    D = Array{Float32}(undef, length.(grid)..., length(grid))
    central∇!(D, V, grid)
    return D
end
function central∇!(D, V, grid)
    h = steps(grid); twoh⁻¹ = inv.(2f0 .* h);
    if any(h .< ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    dimensions = length(grid)
    Δ = makeΔ(dimensions)

    Iₗ, Iᵤ = CartesianIndices(V) |> extrema

    @batch for I in CartesianIndices(V), d in axes(D, dimensions + 1)
        D[I, d] = twoh⁻¹[d] * ( V[min(I + Δ[d], Iᵤ)] - V[max(I - Δ[d], Iₗ)] )
    end

    return D 
end

"""
Given a Vₜ (n₁ × n₂ ... × nₘ) and a drift w (n₁ × n₂ ... × nₘ × 3) returns a matrix D (n₁ × n₂ ... × nₘ × 4), with first three elemnts ∇Vₜ and last ∇Vₜ⋅w.

The finite difference scheme is computed by using second order forward derivatives if the drift is positive and backwards if it is negative.
"""
function dir∇(V, w, grid)
    D = Array{Float32}(undef, length.(grid)..., length(grid) + 1)
    dir∇!(D, V, w, grid)
    return D
end
function dir∇!(D, V, w, grid)
    h = steps(grid); twoh⁻¹ = inv.(2f0 .* h);
    if any(h .< ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    dimensions = length(grid)
    Δ = makeΔ(dimensions)
    
    Iₗ, Iᵤ = CartesianIndices(V) |> extrema
    temp = 0f0

    @batch for I in CartesianIndices(V)
        temp = 0f0

        for (l, Δₗ) ∈ enumerate(Δ)
            D[I, l] = twoh⁻¹[l] * ifelse(w[I, l] > 0,
                -V[min(I + 2Δₗ, Iᵤ)] + 4f0V[min(I + Δₗ, Iᵤ)] - 3f0V[I],
                V[max(I - 2Δₗ, Iₗ)] - 4f0V[max(I - Δₗ, Iₗ)] + 3f0V[I]
            )

            temp += D[I, l] * w[I, l]
        end        
        D[I, dimensions + 1] = temp
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

"""
Given a function G: [0, ∞) × (n₁ × n₂ × n₃)² → (n₁ × n₂ × n₃), with
    ∂ₜ Wₜ = G(t, X, Wₜ), 
a time step h, an evaluation Wₜ, and a time t, computes the Runge-Kutta third order step 
    Δₕ(G) = (h / 8) * (2k₁ + 3k₂ + 3k₃).

And computes 
    Wₜ += Δₕ(G)
"""
function rkstep!(Wₜ, G::Function, t::Float32, X::Array{Float32, 4}; h = 1f-2)
    k₁ = G(t, Wₜ, X)
    k₂ = G(t + (2f0 / 3f0) * h, Wₜ + (2f0 / 3f0) * h * k₁, X)
    k₃ = G(t + (2f0 / 3f0) * h, Wₜ + (2f0 / 3f0) * h * k₂, X)

    Wₜ .+= (h / 8f0) * (2k₁ + 3k₂ + 3k₃)
    return Wₜ
end
function rkstep(Wₜ::AbstractArray{Float32, 3}, G::Function, t::Float32, X::Array{Float32, 4}; h = 1f-2)
    Wₜ₊₁ = copy(Wₜ)
    rkstep!(Wₜ₊₁, G, t, X; h = h)
    return Wₜ₊₁
end