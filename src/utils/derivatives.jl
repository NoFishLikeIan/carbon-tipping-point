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
"5d, gradient sixteen-point stencil in direction w, with dims(w) = (1, 5)"
function S₅¹(w::Matrix{Float32}) 
    # Ugliest function I have ever written
    S = zeros(Float32, d¹, d¹, d¹, d¹, d¹, 1, 1)
    S[:, :, :, :, 1 + d¹ ÷ 2, :, :] .+= (ϵ⁻¹ * w[1]) .* s¹
    S[:, :, :, 1 + d¹ ÷ 2, :, :, :] .+= (ϵ⁻¹ * w[2]) .* s¹
    S[:, :, 1 + d¹ ÷ 2, :, :, :, :] .+= (ϵ⁻¹ * w[3]) .* s¹
    S[:, 1 + d¹ ÷ 2, :, :, :, :, :] .+= (ϵ⁻¹ * w[4]) .* s¹
    S[1 + d¹ ÷ 2, :, :, :, :, :, :] .+= (ϵ⁻¹ * w[5]) .* s¹

    return S
end

baseidx(Δ::Int, a::NTuple{5, Int})::Int = 1 + (a[1] - 1) + (a[2] - 1) * Δ + (a[3] - 1) * Δ^2 + (a[4] - 1) * Δ^3 + (a[5] - 1) * Δ^4

function invbaseidx(Δ::Int, jdx::Int)::NTuple{5, Int}
    i = 1 + mod(jdx - 1, Δ)
    j = 1 + mod((jdx - i) ÷ Δ, Δ)
    k = 1 + mod((jdx - i - (j - 1) * Δ) ÷ (Δ^2), Δ)
    l = 1 + mod((jdx - i - (j - 1) * Δ - (k - 1) * Δ^2) ÷ (Δ^3), Δ)
    m = 1 + mod((jdx - i - (j - 1) * Δ - (k - 1) * Δ^2 - (l - 1) * Δ^3) ÷ (Δ^4), Δ)

    return (i, j, k, l, m)
end

"
Five stencil ∇ₓy in direction dims(w) = (1, 5). Assume y is a (1, n) matrix with entries y = { f(x[i, j, k, l, m]) }ₙ.
"
∇₅(y::Matrix{Float32}, w::Matrix{Float32}) = ∇₅!(similar(y), y, w)
function ∇₅!(D::Matrix{Float32}, y::Matrix{Float32}, w::Matrix{Float32})

    d = size(w, 2)
    n = size(y, 2)

    Δ = floor(Int, n^(1 / d)) # Assumes n = Δ^5; |x| ≡ Δ 

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
