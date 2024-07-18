AbstractModel = Union{TippingModel, JumpModel}

"Emissivity rate implied by abatement `α` at time `t` and carbon concentration `M`"
function ε(t, M, α, model::AbstractModel)
    1. - M * (δₘ(M, model.hogg) + γ(t, model.calibration) - α) / (Gtonoverppm * Eᵇ(t, model.calibration))
end
function ε′(t, M, model::AbstractModel)
    M / (Gtonoverppm * Eᵇ(t, model.calibration))
end

"Drift of dy in the terminal state, t ≥ τ."
bterminal(Xᵢ::Point, args...) = bterminal(Xᵢ.T, args...)
bterminal(T::Float64, χ, model::AbstractModel) = bterminal(T, χ, model.economy, model.damages, model.hogg) 
function bterminal(T::Float64, χ, economy::Economy, damages::GrowthDamages, hogg::Hogg)
    ϕ(economy.τ, χ, economy) - economy.δₖᵖ - d(T, damages, hogg)
end

"Drift of dy."
function b(t, Xᵢ::Point, u::Policy, model::AbstractModel)
    εₜ = ε(t, exp(Xᵢ.m), u.α, model)
    Aₜ = A(t, model.economy)

    abatement = Aₜ * β(t, εₜ, model.economy)

    growth = model.economy.ϱ + ϕ(t, u.χ, model.economy) - model.economy.δₖᵖ
    damage = d(Xᵢ.T, model.damages, model.hogg)

    return growth - abatement - damage
end

"Computes maximum absolute value of the drift of y."
function boundb(t, Xᵢ::Point, model::AbstractModel)
    γₜ = γ(t, model.calibration)
    δₘᵢ = δₘ(exp(Xᵢ.m), model.hogg)

    ll = b(t, Xᵢ, Policy(0., 0.), model)
    lr = b(t, Xᵢ, Policy(0., γₜ + δₘᵢ), model)
    rl = b(t, Xᵢ, Policy(1., 0.), model)
    rr = b(t, Xᵢ, Policy(1., γₜ + δₘᵢ), model)

    return max(abs(ll), abs(lr), abs(rl), abs(rr))
end