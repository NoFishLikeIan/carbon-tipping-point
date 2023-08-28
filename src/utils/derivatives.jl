using NNlib: conv
using LoopVectorization: @tturbo, @turbo

const ϵ = cbrt(eps(Float32))
const ϵ⁻¹ = inv(ϵ)
const ϵ⁻² = inv(ϵ^2)

function paddedrange(from::Float32, to::Float32; pad = 2ϵ)
    collect(range(from - pad, to + pad; step = ϵ)[:, :]')
end

const s¹ = [-1f0, 8f0, 0f0, -8f0, 1f0] ./ 12f0 # stencil of first derivative
const d¹ = length(s¹)
const s² = [1f0, -2f0, 1f0] # stencil of second derivative

# One dimension
const S₁¹ = reshape(ϵ⁻¹ .* s¹, 1, length(s¹), 1, 1)
const S₁² = reshape(ϵ⁻² .* s², 1, length(s²), 1, 1)

function ∂(y)
    dropdims(conv(reshape(y, 1, size(y, 2), 1, 1), S₁¹), dims = (3, 4))
end

function ∂²(y)
    dropdims(conv(reshape(y, 1, size(y, 2), 1, 1), S₁²), dims = (3, 4))
end

# Two dimensions
"2d, gradient sixteen-point stencil in direction w, with dims(w) = (1, 2)"
function S₂¹(w::Matrix{Float32}) 
    S = zeros(Float32, d¹, d¹, 1, 1)
    S[3, :, :, :] .+= (ϵ⁻¹ * w[1]) .* s¹
    S[:, 3, :, :] .+= (ϵ⁻¹ * w[2]) .* s¹

    return S
end

"Gradient ∇ₓy in direction dims(w) = (1, 2)"
function ∇₂(y::Matrix{Float32}, w::Matrix{Float32})
    dropdims(
        conv(reshape(y, size(y, 1), size(y, 2), 1, 1), S₂¹(w)),
        dims = (3, 4)
    )
end


# Five dimensions
"Five stencil ∇ₓy in direction dims(w) = (1, 5). Assume y is a (1, n) matrix with entries y = { f(x[i, j, k, l, m]) }ₙ."
∇₅(y::Matrix{Float32}, w::Matrix{Float32}) = ∇₅!(similar(y), y, w)
function ∇₅!(D::Matrix{Float32}, y::Matrix{Float32}, w::Matrix{Float32})
    Δ = floor(Int, size(y, 2)^(1 / 5)) # Assumes n = Δ^5; |x| ≡ Δ 

    Δ², Δ³, Δ⁴ = Δ^2, Δ^3, Δ^4
    
    @tturbo for jdx in axes(D, 2)   
        i = 1 + mod(jdx - 1, Δ)
        j = 1 + mod((jdx - i) ÷ Δ, Δ)
        k = 1 + mod((jdx - i - (j - 1) * Δ) ÷ (Δ²), Δ)
        l = 1 + mod((jdx - i - (j - 1) * Δ - (k - 1) * Δ²) ÷ (Δ³), Δ)
        m = 1 + mod((jdx - i - (j - 1) * Δ - (k - 1) * Δ² - (l - 1) * Δ³) ÷ Δ⁴, Δ)

        jklm = jdx - (i - 1)
        iklm = jdx - (j - 1) * Δ
        ijlm = jdx - (k - 1) * Δ²
        ijkm = jdx - (l - 1) * Δ³
        ijkl = jdx - (m - 1) * Δ⁴

    
        D[jdx] = (ϵ⁻¹ / 12f0) * (  
            w[1] * (   
                8f0 * (
                    y[jklm + min(i + 1, Δ) - 1] -
                    y[jklm + max(i - 1, 1) - 1]
                ) - (
                    y[jklm + min(i + 2, Δ) - 1] - 
                    y[jklm + max(i - 2, 1) - 1]
                )
            ) + w[2] * (   
                8f0 * (
                    y[iklm + Δ * (min(j + 1, Δ) - 1)] -
                    y[iklm + Δ * (max(j - 1, 1) - 1)]
                ) - (
                    y[iklm + Δ * (min(j + 2, Δ) - 1)] - 
                    y[iklm + Δ * (max(j - 2, 1) - 1)]
                )
            ) + w[3] * (   
                8f0 * (
                    y[ijlm + Δ² * (min(k + 1, Δ) - 1)] -
                    y[ijlm + Δ² * (max(k - 1, 1) - 1)]
                ) - (
                    y[ijlm + Δ² * (min(k + 2, Δ) - 1)] - 
                    y[ijlm + Δ² * (max(k - 2, 1) - 1)]
                )
            ) + w[4] * (   
                8f0 * (
                    y[ijkm + Δ³ * (min(l + 1, Δ) - 1)] -
                    y[ijkm + Δ³ * (max(l - 1, 1) - 1)]
                ) - (
                    y[ijkm + Δ³ * (min(l + 2, Δ) - 1)] - 
                    y[ijkm + Δ³ * (max(l - 2, 1) - 1)]
                )
            )  + w[5] *  (   
                8f0 * (
                    y[ijkl + Δ⁴ * (min(m + 1, Δ) - 1)] -
                    y[ijkl + Δ⁴ * (max(m - 1, 1) - 1)]
                ) - (
                    y[ijkl + Δ⁴ * (min(m + 2, Δ) - 1)] - 
                    y[ijkl + Δ⁴ * (max(m - 2, 1) - 1)]
                )
            )
        )
    end

    return D 
end

# ---- Ad hoc functions ----

"
Takes a row major V (1 × n) and three directions w (1 × 5), Fα (1 × 2), and Fχ (1 × 1). 

Computes the three necessary gradients:
1. HJB: ∇V⋅w
2. FOC α: ∇V₃₄⋅Fα
3. FOC χ: ∇V₄⋅Fχ

The output is a row major directional derivative matrix D (3 × n).
"
∇V′w(V::Matrix{Float32}, w::Matrix{Float32}, Fα::Matrix{Float32}, Fχ::Matrix{Float32}) = ∇V′w!(Matrix{Float32}(undef, 3, size(V, 2)), V, w, Fα, Fχ)
function ∇V′w!(D::Matrix{Float32}, V::Matrix{Float32}, w::Matrix{Float32}, Fα::Matrix{Float32}, Fχ::Matrix{Float32})::Matrix{Float32}
    Δ = floor(Int, size(V, 2)^(1 / 5)) # Assumes the grid for V is Δ^5; |x| ≡ Δ

    Δ², Δ³, Δ⁴ = Δ^2, Δ^3, Δ^4
    
    # Iteration for w
    @tturbo for jdx in axes(D, 2)   
        i = 1 + mod(jdx - 1, Δ)
        j = 1 + mod((jdx - i) ÷ Δ, Δ)
        k = 1 + mod((jdx - i - (j - 1) * Δ) ÷ (Δ²), Δ)
        l = 1 + mod((jdx - i - (j - 1) * Δ - (k - 1) * Δ²) ÷ (Δ³), Δ)
        m = 1 + mod((jdx - i - (j - 1) * Δ - (k - 1) * Δ² - (l - 1) * Δ³) ÷ Δ⁴, Δ)

        jklm = jdx - (i - 1)
        iklm = jdx - (j - 1) * Δ
        ijlm = jdx - (k - 1) * Δ²
        ijkm = jdx - (l - 1) * Δ³
        ijkl = jdx - (m - 1) * Δ⁴

        ∂₃ = 8f0 * (
            V[ijlm + Δ² * (min(k + 1, Δ) - 1)] -
            V[ijlm + Δ² * (max(k - 1, 1) - 1)]
        ) - (
            V[ijlm + Δ² * (min(k + 2, Δ) - 1)] - 
            V[ijlm + Δ² * (max(k - 2, 1) - 1)]
        )

        ∂₄ = 8f0 * (
            V[ijkm + Δ³ * (min(l + 1, Δ) - 1)] -
            V[ijkm + Δ³ * (max(l - 1, 1) - 1)]
        ) - (
            V[ijkm + Δ³ * (min(l + 2, Δ) - 1)] - 
            V[ijkm + Δ³ * (max(l - 2, 1) - 1)]
        )

        # ∇V⋅w
        D[1, jdx] = (ϵ⁻¹ / 12f0) * (
            # Dimension T  
            w[1] * (   
                8f0 * (
                    V[jklm + min(i + 1, Δ) - 1] -
                    V[jklm + max(i - 1, 1) - 1]
                ) - (
                    V[jklm + min(i + 2, Δ) - 1] - 
                    V[jklm + max(i - 2, 1) - 1]
                )
            ) + 
            # Dimension N
            w[2] * (   
                8f0 * (
                    V[iklm + Δ * (min(j + 1, Δ) - 1)] -
                    V[iklm + Δ * (max(j - 1, 1) - 1)]
                ) - (
                    V[iklm + Δ * (min(j + 2, Δ) - 1)] - 
                    V[iklm + Δ * (max(j - 2, 1) - 1)]
                )
            ) + 
            # Dimension m
            w[3] * ∂₃ + 
            # Dimension y
            w[4] * ∂₄ +  
            # Dimension t, use forward difference
            w[5] *  (   
                24f0V[ijkl + Δ⁴ * (min(m + 1, Δ) - 1)] - 
                6f0V[ijkl + Δ⁴ * (min(m + 2, Δ) - 1)] -
                18f0V[jdx]
            )
        )

        D[2, jdx] = (ϵ⁻¹ / 12f0) * (∂₃ * Fα[1] + ∂₄ * Fα[2]) # ∇V₃₄⋅Fα
        D[3, jdx] = (ϵ⁻¹ / 12f0) * ∂₄ * Fχ[1] # ∇V₄ ⋅ Fχ
    end

    return D
end

"Compute second derivative of y in direction of the first input x₁:
( ∂²f(x₁, x₂, x₃, x₄, x₅) / ∂x₁² )"
∂²₁(y::Matrix{Float32}) = ∂²₁!(similar(y), y)
function ∂²₁!(D::Matrix{Float32}, y::Matrix{Float32})
    Δ = floor(Int, size(y, 2)^(1 / 5)) # Assumes n = Δ^5; |x| ≡ Δ 
    
    D .= (-2f0 * ϵ⁻²) .* y
    
    @tturbo for jdx in axes(D, 2)   
        i = 1 + mod(jdx - 1, Δ)
        jklm = jdx - (i - 1)

        D[jdx] += ϵ⁻² * (
            y[jklm + min(i + 1, Δ) - 1] + y[jklm + max(i - 1, 1) - 1]
        )
    end

    return D 
end