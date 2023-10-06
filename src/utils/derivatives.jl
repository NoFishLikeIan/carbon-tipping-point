const ϵ = cbrt(eps(Float32))
const Δi = CartesianIndex(1, 0, 0);
const Δj = CartesianIndex(0, 1, 0);
const Δk = CartesianIndex(0, 0, 1);
const Δ = (Δi, Δj, Δk);

"""
Upwind-downind directional derivative of V at idx along direction Δd
"""
function ∂(idx::CartesianIndex, V::Array{Float32, 3}, ispos::Bool, Δd::CartesianIndex, Rₑ::Tuple{CartesianIndex, CartesianIndex})
    if ispos > 0
        -V[min(idx + 2Δd, Rₑ[2])] + 4f0V[min(idx + Δd, Rₑ[2])] - 3f0V[idx]
    else
        V[min(idx - 2Δd, Rₑ[1])] - 4f0V[min(idx - Δd, Rₑ[1])] + 3f0V[idx]
    end
end

"""
Given a Vₜ (n₁ × n₂ × n₃) and a drift w (n₁ × n₂ × n₃ × 3) returns a matrix D (n₁ × n₂ × n₃ × 4), with first three elemnts ∇Vₜ and last ∇Vₜ⋅w.

The finite difference scheme is computed by using second order forward derivatives if the drift is positive and backwards if it is negative.
"""
function dir∇(V::Array{Float32, 3}, w::Array{Float32, 4}, grid::StateRegularGrid)    
    D = Array{Float32}(undef, size(V)..., length(grid) + 1)
    dir∇!(D, V, w, grid)
    return D
end
function dir∇!(D::Array{Float32, 4}, V::Array{Float32, 3}, w::Array{Float32, 4}, grid::StateRegularGrid)
    h = steps(grid)

    if any(h .< ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end
    twoh⁻¹ = inv.(2f0 .* h)
    
    R = CartesianIndices(V)
    R₁, Rₙ = extrema(R)

    # TODO: vectorize
    for idx in R
        inner = 0f0

        for (l, Δₗ) ∈ enumerate(Δ)
            D[idx, l] = twoh⁻¹[l] * (w[idx, l] > 0 ?
                -V[min(idx + 2Δₗ, Rₙ)] + 4f0V[min(idx + Δₗ, Rₙ)] - 3f0V[idx] :
                V[max(idx - 2Δₗ, R₁)] - 4f0V[max(idx - Δₗ, R₁)] + 3f0V[idx]
            )

            inner += D[idx, l] * w[idx, l]
        end

        
        D[idx, 4] = inner
    end

    return D 
end

"""
Given a Vₜ (n₁ × n₂ × n₃) returns a matrix D (n₁ × n₂ × n₃ × 3), with elements ∇Vₜ and last ∇Vₜ⋅w.
"""
function central∇(V::Array{Float32, 3}, grid::StateRegularGrid)
    D = Array{Float32}(undef, size(V, 1), size(V, 2), size(V, 3), length(grid))
    central∇!(D, V, grid)
    return D
end
function central∇!(D::Array{Float32, 4}, V::Array{Float32, 3}, grid::StateRegularGrid)
    h = steps(grid)

    if any(h .< ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    twoh⁻¹ = inv.(2f0 .* h)

    R = CartesianIndices(V)
    R₁, Rₙ = extrema(R)

    # TODO: vectorize
    for idx in R
        D[idx, 1] = twoh⁻¹[1] * ( V[min(idx + Δi, Rₙ)] - V[max(idx - Δi, R₁)] )
        D[idx, 2] = twoh⁻¹[2] * ( V[min(idx + Δj, Rₙ)] - V[max(idx - Δj, R₁)] )
        D[idx, 3] = twoh⁻¹[3] * ( V[min(idx + Δk, Rₙ)] - V[max(idx - Δk, R₁)] )
    end

    return D 
end

"""
Given a Vₜ (n₁ × n₂ × n₃) computes the second derivative in the direction of the l-th input xₗ.
"""
function ∂²(l::Int, V::Array{Float32, 3}, grid::StateRegularGrid)
    D² = similar(V)
    ∂²!(D², l, V, grid)
    return D²
end
function ∂²!(D²::Array{Float32, 3}, l::Int, V::Array{Float32, 3}, grid::StateRegularGrid)
    hₗ = steps(grid)[l]

    if (hₗ < ϵ) @warn "Step size smaller than machine ϵ ≈ 4.9e-3" end

    hₗ⁻² = inv(hₗ^2)

    R = CartesianIndices(V)
    R₁, Rₙ = extrema(R)

    # TODO: vectorize
    for idx in R
        D²[idx] = hₗ⁻² * (
            V[min(idx + Δ[l], Rₙ)] - 2f0 * V[idx] + V[max(idx - Δ[l], R₁)]
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