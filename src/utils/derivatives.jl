const ϵ = cbrt(eps(Float32))
const ϵ⁻¹ = inv(ϵ)
const ϵ⁻² = inv(ϵ^2)

function paddedrange(from::Float32, to::Float32; pad = 2ϵ)
    collect(range(from - pad, to + pad; step = ϵ)[:, :]')
end

function ∂₊(V, idx, onestep, twostep)
    (ϵ⁻¹ / 2f0) .* (-V[twostep] + 4f0V[onestep] - 3f0V[idx])
end

function ∂₋(V, idx, onestep, twostep)
    (ϵ⁻¹ / 2f0) .* (3f0V[idx] - 4f0V[onestep] + V[twostep])
end

"""
Given a Vₜ (1 × n)  and a drift w (3 × n) returns a matrix ∇Vₜ⋅w (1 × n). Note that Vₜ_ijk = Vₜ(T, m, y).

The finite difference scheme is computed by using second order forward derivatives if the drift is positive and backwards if it is negative.

Accepts δ (3 × 1) which is the domain size. It defaults to δ = 1.
"""
function dir∇V(V::Matrix{Float32}, w::Matrix{Float32})    
    D = Matrix{Float32}(undef, size(w))
    return dir∇V!(D, V, w)
end
function dir∇V!(D::Matrix{Float32}, V::Matrix{Float32}, w::Matrix{Float32})
    m, n = size(w)
    Δ = round(Int, n^(1 / m)) # Assumes n = Δ^m where |x| ≡ Δ 
    Δ² = Δ^2
 
    posdrift = w .> 0
    
    @inbounds for idx in axes(D, 2)  
        i = 1 + mod(idx - 1, Δ)
        j = 1 + mod((idx - i) ÷ Δ, Δ)
        k = 1 + mod((idx - i - (j - 1) * Δ) ÷ (Δ²), Δ)

        jk = idx - (i - 1)
        ik = idx - (j - 1) * Δ
        ij = idx - (k - 1) * Δ²


        D[1, idx] = posdrift[1, idx] ? # ∂T
                ∂₊(V, idx, jk + (min(i + 1, Δ) - 1), jk + (min(i + 2, Δ) - 1)) :
                ∂₋(V, idx, jk + (max(i - 1, 1) - 1), jk + (max(i - 2, 1) - 1))

        D[2, idx] = posdrift[2, idx] ? # ∂m
                ∂₊(V, idx, ik + Δ * (min(j + 1, Δ) - 1), ik + Δ * (min(j + 2, Δ) - 1)) :
                ∂₋(V, idx, ik + Δ * (max(j - 1, 1) - 1), ik + Δ * (max(j - 2, 1) - 1))

        D[3, idx] = posdrift[3, idx] ? # ∂y
                ∂₊(V, idx, ij + Δ² * (min(k + 1, Δ) - 1), ij + Δ² * (min(k + 2, Δ) - 1)) :
                ∂₋(V, idx, ij + Δ² * (max(k - 1, 1) - 1), ij + Δ² * (max(k - 2, 1) - 1)) 
    end

    return D 
end

"""
Given a V (1 × n) returns a matrix D (3 × n) with rows (∂TV, ∂mV, ∂yV) where and (∂TV, ∂mV, ∂yV) are computed via central differences. Note that V_ijk = V(T, m, y)
"""
function central∇V(V::Matrix{Float32})    
    central∇V!(Matrix{Float32}(undef, 3, size(V, 2)), V)
end
function central∇V!(D::Matrix{Float32}, V::Matrix{Float32})
    dims, n = size(D)
    Δ = round(Int, n^(1 / dims)) # Assumes n = Δ^dims; |x| ≡ Δ 
    Δ² = Δ^2

    @inbounds for idx in axes(D, 2)  
        i = 1 + mod(idx - 1, Δ)
        j = 1 + mod((idx - i) ÷ Δ, Δ)
        k = 1 + mod((idx - i - (j - 1) * Δ) ÷ (Δ²), Δ)

        jk = idx - (i - 1)
        ik = idx - (j - 1) * Δ
        ij = idx - (k - 1) * Δ²

        D[1, idx] = (ϵ⁻¹ / 2f0) .* (V[jk + (min(i + 1, Δ) - 1)] - V[jk + (max(i - 1, 1) - 1)])

        D[2, idx] = (ϵ⁻¹ / 2f0) .* (V[ik + Δ * (min(j + 1, Δ) - 1)] - V[ik + Δ * (max(j - 1, 1) - 1)])

        D[3, idx] = (ϵ⁻¹ / 2f0) .* (V[ij + Δ² * (min(k + 1, Δ) - 1)] - V[ij + Δ² * (max(k - 1, 1) - 1)])
    end

    return D 
end

"Compute second derivative of y in direction of the first input x₁:
( ∂²/(∂x₁)² f(x₁, x₂, x₃) )"
function ∂²T(V::Matrix{Float32}; m = 3)
    D = similar(V)
    ∂²T!(D, V; m = m)
end
function ∂²T!(D::Matrix{Float32}, V::Matrix{Float32}; m = 3)
    Δ = round(Int, size(V, 2)^(1 / m)) # Assumes n = Δ^m; |x| ≡ Δ 
        
    @inbounds for idx in axes(D, 2) 
        i = 1 + mod(idx - 1, Δ)
        jk = idx - (i - 1)

        D[idx] = ϵ⁻² * (
            V[jk + (min(i + 1, Δ) - 1)] + 
            V[jk + (max(i - 1, 1) - 1)] - 
            2f0 * V[idx]
        )
    end

    return D 
end