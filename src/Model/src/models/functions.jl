AbstractModel{D, P} = Union{TippingModel{D, P}, JumpModel{D, P}} where {D <: Damages, P <: Preferences}

"Emissivity rate implied by abatement `α` at time `t` and carbon concentration `M`"
function ε(t, M, α, model::AbstractModel)
    1. - M * (δₘ(M, model.hogg) + γ(t, model.calibration) - α) / (Gtonoverppm * Eᵇ(t, model.calibration))
end
function ε′(t, M, model::AbstractModel)
    M / (Gtonoverppm * Eᵇ(t, model.calibration))
end

"Drift of log output y in the terminal state, t ≥ τ"
function bterminal(χ, model::AbstractModel{LevelDamages, P}) where P <: Preferences
    growth = model.economy.ϱ - model.economy.δₖᵖ
    investments = ϕ(model.economy.τ, χ, model.economy)

    return growth + investments
end
function bterminal(T::Float64, χ, model::AbstractModel{GrowthDamages, P}) where P <: Preferences
    growth = model.economy.ϱ - model.economy.δₖᵖ
    investments = ϕ(model.economy.τ, χ, model.economy)
    damage = d(T, model.damages, model.hogg)

    return growth + investments - damage
end

"Drift of log output y for t < τ"
function b(t, Xᵢ::Point, u::Policy, model::AbstractModel{GrowthDamages, P}) where P <: Preferences
    εₜ = ε(t, exp(Xᵢ.m), u.α, model)
    Aₜ = A(t, model.economy)

    abatement = Aₜ * β(t, εₜ, model.economy)

    growth = model.economy.ϱ - model.economy.δₖᵖ
    investments = ϕ(t, u.χ, model.economy)
    damage = d(Xᵢ.T, model.damages, model.hogg)

    return growth + investments - abatement - damage
end
function b(t, Xᵢ::Point, u::Policy, model::AbstractModel{LevelDamages, P}) where P <: Preferences
    εₜ = ε(t, exp(Xᵢ.m), u.α, model)
    Aₜ = A(t, model.economy)

    abatement = Aₜ * β(t, εₜ, model.economy)

    growth = model.economy.ϱ - model.economy.δₖᵖ
    investments = ϕ(t, u.χ, model.economy)

    return growth + investments - abatement
end

"Computes maximum absolute value of the drift of output y."
function bbound(t, Xᵢ::Point, model::AbstractModel)
    γₜ = γ(t, model.calibration)
    δₘᵢ = δₘ(exp(Xᵢ.m), model.hogg)

    ll = b(t, Xᵢ, Policy(0., 0.), model)
    lr = b(t, Xᵢ, Policy(0., γₜ + δₘᵢ), model)
    rl = b(t, Xᵢ, Policy(1., 0.), model)
    rr = b(t, Xᵢ, Policy(1., γₜ + δₘᵢ), model)

    return max(abs(ll), abs(lr), abs(rl), abs(rr))
end