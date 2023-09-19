const ϵ = cbrt(eps(Float32))
const ϵ⁻¹ = inv(ϵ)
const ϵ⁻² = inv(ϵ^2)

function paddedrange(from::Float32, to::Float32; pad = 2ϵ)
    collect(range(from - pad, to + pad; step = ϵ)[:, :]')
end

"""
Given a V (1 × n)  and a drift w (3 × n) returns a matrix D (4 × n) with rows:
- D₁ = ∂tV always computed forward
- D₂ = ∂mV
- D₃ = ∂yV
- D₄ = ∇V⋅w

The finite difference scheme is computed by using second order forward derivatives if the drift is positive and backwards if it is negative

Note that V_tijk ≈ V(t, T, m, y)
"""
function dir∇V(V::Matrix{Float32}, w::Matrix{Float32})    
    D = Matrix{Float32}(undef, 4, size(V, 2))
    return dir∇V!(D, V, w)
end
function dir∇V!(D::Matrix{Float32}, V::Matrix{Float32}, w::Matrix{Float32})
    Δ = floor(Int, size(V, 2)^(1 / 4)) # Assumes n = Δ^4; |x| ≡ Δ 

    Δ², Δ³ = Δ^2, Δ^3

    posdrift = w .> 0
    
    for index in axes(D, 2)  
        t = 1 + mod(index - 1, Δ)
        i = 1 + mod((index - t) ÷ Δ, Δ)
        j = 1 + mod((index - t - (i - 1) * Δ) ÷ (Δ²), Δ)
        k = 1 + mod((index - t - (i - 1) * Δ - (j - 1) * Δ²) ÷ (Δ³), Δ)

        jkl = index - (t - 1)
        tjk = index - (i - 1) * Δ
        tik = index - (j - 1) * Δ²
        tij = index - (k - 1) * Δ³

        D[1, index] = ϵ⁻¹ .* (V[jkl + min(t + 1, Δ) - 1] - V[index]) # ∂t

        D[2, index] = (ϵ⁻¹ / 2f0) .* (posdrift[2, index] ? 
            -V[tik + Δ² * (min(j + 2, Δ) - 1)] + 4f0V[tik + Δ² * (min(j + 1, Δ) - 1)] - 3f0V[index] :
            3f0V[index] - 4f0V[tik + Δ² * (max(j - 1, 1) - 1)] - V[tik + Δ² * (max(j - 2, 1) - 1)]
        ) # ∂m
     
        D[3, index] = (ϵ⁻¹ / 2f0) .* (posdrift[3, index] ? 
            -V[tij + Δ³ * (min(k + 2, Δ) - 1)] + 4f0V[tij + Δ³ * (min(k + 1, Δ) - 1)] - 3f0V[index] :
            3f0V[index] - 4f0V[tij + Δ³ * (max(k - 1, 1) - 1)] - V[tij + Δ³ * (max(k - 2, 1) - 1)]
        ) # ∂y
    
        D[4, index] = 
            w[2, index] * D[2, index] + 
            w[3, index] * D[3, index] + 
            (ϵ⁻¹ / 2f0) .* (posdrift[2, index] ? 
                -V[tjk + Δ * (min(i + 2, Δ) - 1)] + 4f0V[tjk + Δ * (min(i + 1, Δ) - 1)] - 3f0V[index] :
                3f0V[index] - 4f0V[tjk + Δ * (max(i - 1, 1) - 1)] - V[tjk + Δ * (max(i - 2, 1) - 1)]
            ) # ∂T
    end

    return D 
end

"Compute second derivative of y in direction of the first input x₁:
( ∂²f(x₁, x₂, x₃, x₄) / ∂x₂² )"
function ∂²T(y)
    D = similar(y)
    ∂²T!(D, y)
    return copy(D)
end
function ∂²T!(D, y)
    Δ = floor(Int, size(y, 2)^(1 / 4)) # Assumes n = Δ^4; |x| ≡ Δ 
        
    for index in axes(D, 2) 
        t = 1 + mod(index - 1, Δ)
        i = 1 + mod((index - t) ÷ Δ, Δ)
        tjk = index - (i - 1) * Δ

        D[index] = ϵ⁻² * (
            y[tjk + Δ * (min(i + 1, Δ) - 1)] + y[tjk + Δ * (max(i - 1, 1) - 1)] - 2f0 * y[index]
        )
    end

    return D 
end