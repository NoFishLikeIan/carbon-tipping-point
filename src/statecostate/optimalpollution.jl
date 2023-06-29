"""Policy function"""
function φ(λₘ, x, economy::LinearQuadratic)
    @unpack ē, β₁, β₀ = economy

    return clamp((β₀ + λₘ) / β₁, 0, ē)
end

function φ(λₘ, x, economy::Ramsey)    
    h(e) = ∂ₑc(e, x, economy) + λₘ * c(e, x, economy)

    if h(economy.ē) > 0 return economy.ē end
    if h(0) < 0 return 0 end

    return find_zero(h, (0, economy.ē))
end

function H(x, m, λₓ, λₘ, climate::Climate, economy::EconomicModel)
	e = φ(λₘ, x, economy)
	return u(e, x, economy) + λₓ * μ(x, m, climate) + λₘ * (e - climate.δ * m)
end
