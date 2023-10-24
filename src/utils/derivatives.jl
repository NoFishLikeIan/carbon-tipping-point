using Polyester: @batch

const ϵ = cbrt(eps(Float32));
const Δi = CartesianIndex(1, 0, 0);
const Δj = CartesianIndex(0, 1, 0);
const Δk = CartesianIndex(0, 0, 1);
const Δ = (Δi, Δj, Δk);

"""
Given a Vₜ (n₁ × n₂ × n₃) returns a matrix D (n₁ × n₂ × n₃ × 3), with elements ∇Vₜ and last ∇Vₜ⋅w.
"""
function central∇(V, grid)
    D = Array{Float32}(undef, length.(grid)..., length(grid))
    central∇!(D, V, grid)
    return D
end
function central∇!(D, V, grid)
    h = steps(grid)
    dimensions = length(grid)

    if any(h .< ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    twoh⁻¹ = inv.(2f0 .* h)

    R = CartesianIndices(V)
    R₁, Rₙ = extrema(R)

    @batch for I in R, d in axes(D, dimensions + 1)
        D[I, d] = twoh⁻¹[d] * ( V[min(I + Δ[d], Rₙ)] - V[max(I - Δ[d], R₁)] )
    end

    return D 
end

"""
Given a Vₜ (n₁ × n₂ × n₃) and a drift w (n₁ × n₂ × n₃ × 3) returns a matrix D (n₁ × n₂ × n₃ × 4), with first three elemnts ∇Vₜ and last ∇Vₜ⋅w.

The finite difference scheme is computed by using second order forward derivatives if the drift is positive and backwards if it is negative.
"""
function dir∇(V::FieldGrid, w, grid)::VectorGrid  
    D = Array{Float32}(undef, length.(grid)..., length(grid) + 1)
    dir∇!(D, V, w, grid)
    return D
end
function dir∇!(D, V, w, grid)
    h = steps(grid)

    if any(h .< ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end
    twoh⁻¹ = inv.(2f0 .* h)
    
    R = CartesianIndices(V)
    R₁, Rₙ = extrema(R)
    temp = 0f0

    @batch for I in R
        temp = 0f0

        for (l, Δₗ) ∈ enumerate(Δ)
            D[I, l] = twoh⁻¹[l] * ifelse(w[I, l] > 0,
                -V[min(I + 2Δₗ, Rₙ)] + 4f0V[min(I + Δₗ, Rₙ)] - 3f0V[I],
                V[max(I - 2Δₗ, R₁)] - 4f0V[max(I - Δₗ, R₁)] + 3f0V[I]
            )

            temp += D[I, l] * w[I, l]
        end

        
        D[I, 4] = temp
    end

    return D 
end

"""
Given a Vₜ (n₁ × n₂ × n₃) computes the second derivative in the direction of the l-th input xₗ.
"""
function ∂²(V::FieldGrid, grid; dim = 1)::FieldGrid
    D² = similar(V)
    ∂²!(D², V, grid; dim = dim)
    return D²
end
function ∂²!(D², V, grid; dim = 1)
    hₗ = steps(grid)[dim]

    if (hₗ < ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    hₗ⁻² = inv(hₗ^2)

    R = CartesianIndices(V)
    R₁, Rₙ = extrema(R)
    
    @batch for idx in R
        D²[idx] = hₗ⁻² * (
            V[min(idx + Δ[dim], Rₙ)] - 2f0 * V[idx] + V[max(idx - Δ[dim], R₁)]
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
function rkstep!(Wₜ::FieldGrid, G::Function, t::Float32, X::Array{Float32, 4}; h = 1f-2)
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