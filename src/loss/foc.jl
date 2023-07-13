function f′(c, u, economy::Economy)
    @unpack ρ, θ, ψ = economy
    ψ⁻¹ = 1 / ψ

    ucont = ((1 - θ) * u)^((1 - θ) / (1 - ψ⁻¹))

    return ρ * (1 - θ) * u * c^(-ψ⁻¹) / ucont
end

function ϕ′(t, χ, economy::Economy)
    Aₜ = A(t, economy)

    return economy.κ * Aₜ^2 * (1 - χ) - Aₜ
end