# TODO: break up these functions
AbstractModel{D, P} = Union{TippingModel{D, P}, JumpModel{D, P}} where {D <: Damages, P <: Preferences}

μ(T, m, model::TippingModel) = μ(T, m, model.hogg, model.albedo)
μ(T, m, model::JumpModel) = μ(T, m, model.hogg)
mstable(T, model::TippingModel) = mstable(T, model.hogg, model.albedo)
mstable(T, model::JumpModel) = mstable(T, model.hogg)

"Drift of log output y in the terminal state, t ≥ τ"
function bterminal(χ, model::AbstractModel{LevelDamages, P}) where P <: Preferences
    growth = model.economy.ϱ - model.economy.δₖᵖ
    investments = ϕ(model.economy.τ, χ, model.economy)

    return growth + investments
end
function bterminal(_, χ, model::AbstractModel{LevelDamages, P}) where P <: Preferences # This is kept for compatibility, TODO: There is probably a better way to do it.
    bterminal(χ, model)
end

function bterminal(T::Float64, χ, model::AbstractModel{GrowthDamages, P}) where P <: Preferences
    growth = model.economy.ϱ - model.economy.δₖᵖ
    investments = ϕ(model.economy.τ, χ, model.economy)
    damage = d(T, model.damages, model.hogg)

    return growth + investments - damage
end

"Emissivity rate implied by abatement `α` at time `t` and carbon concentration `M`"
function ε(t, M, α, model::AbstractModel)
    α / (δₘ(M, model.hogg) + γ(t, model.calibration))
end

"Drift of log output y for t < τ" # TODO: Combine the two drifts
function b(t, Xᵢ::Point, u::Policy, model::AbstractModel{GrowthDamages, P}) where P <: Preferences
    εₜ = ε(t, exp(Xᵢ.m), u.α, model)
    Aₜ = A(t, model.economy)

    abatement = Aₜ * β(t, εₜ, model.economy)

    growth = model.economy.ϱ - model.economy.δₖᵖ
    investments = ϕ(t, u.χ, model.economy)
    damage = d(Xᵢ.T, model.damages, model.hogg)

    return growth + investments - abatement - damage
end
function b(t, Xᵢ::Point, u::Policy, model::AbstractModel)
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

function terminaloutputfct(Tᵢ, Δt, χ, model::AbstractModel)
    drift = bterminal(Tᵢ, χ, model) - model.preferences.θ * model.economy.σₖ^2 / 2
     
    adj = Δt * (1 - model.preferences.θ) * drift

    return max(1 + adj, 0.)
end

function outputfct(t, Xᵢ::Point, Δt, u::Policy, model::AbstractModel)
    drift = b(t, Xᵢ, u, model) - model.preferences.θ * model.economy.σₖ^2 / 2

    adj = Δt * (1 - model.preferences.θ) * drift

    return max(1 + adj, 0.)
end

"Computes the temperature level for which it is impossible to achieve positive GDP growth"
function criticaltemperature(model::AbstractModel{GrowthDamages, P}) where P <: Preferences
    maximumgrowth = Tᵢ -> begin
        rate, _ = gss(χ -> bterminal(Tᵢ, χ, model), 0., 1.)
        return rate
    end

    find_zero(maximumgrowth, model.hogg.Tᵖ .+ (0., 15.))
end

function constructdefaultgrid(N, model::AbstractModel)
    T̄ = typeof(model.damages) <: LevelDamages ? 
    model.hogg.T₀ + 8. : criticaltemperature(model)

    Tdomain = (model.hogg.Tᵖ, T̄)
    mdomain = (
        mstable(Tdomain[1], model), 
        mstable(Tdomain[2], model)
    )

    RegularGrid([Tdomain, mdomain], N)
end