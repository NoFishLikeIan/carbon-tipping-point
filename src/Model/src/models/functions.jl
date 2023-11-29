function b(t, Xᵢ::Point, policy::Policy, model::ModelInstance)
    @unpack economy, hogg, albedo, calibration = model

    εₜ = ε(t, exp(Xᵢ.m), policy.α, model)
    Aₜ = A(t, economy)

    abatement = Aₜ * β(t, εₜ, economy)

    economy.ϱ + ϕ(t, policy.χ, economy) - economy.δₖᵖ - abatement - d(Xᵢ.T, economy, hogg)
end

"Emissivity rate implied by abatement `α` at time `t` and carbon concentration `M`"
function ε(t, M, α, model::ModelInstance)
    1f0 - M * (δₘ(M, model.hogg) + γ(t, model.economy, model.calibration) - α) / (Gtonoverppm * Eᵇ(t, model.economy, model.calibration))
end
function ε′(t, M, model::ModelInstance)
    M / (Gtonoverppm * Eᵇ(t, model.economy, model.calibration))
end

"Drift of the state space"
function drift(t, Xᵢ::Point, policy::Policy, model::ModelInstance)
    @unpack economy, hogg, albedo, calibration = model

    abatement = A(t, economy) * β(t, ε(t, exp(Xᵢ.m), policy.α, model), economy)
    
    Drift(
        μ(Xᵢ.T, Xᵢ.m, hogg, albedo) / model.hogg.ϵ,
        γ(t, economy, calibration) - policy.α,
        economy.ϱ - economy.δₖᵖ + ϕ(t, policy.χ, economy) - abatement - d(Xᵢ.T, economy, hogg)
    )
end

"Maximum absolute drift of the state space"
function bounddrift(t, Xᵢ::Point, model::ModelInstance)
    @unpack economy, hogg, albedo, calibration = model
    
    Drift(
        abs(μ(Xᵢ.T, Xᵢ.m, hogg, albedo) / model.hogg.ϵ),
        γ(t, economy, calibration),
        abs(economy.ϱ - economy.δₖᵖ - d(Xᵢ.T, economy, hogg) + ϕ(t, 0f0, economy))
    )
end

function var(model::ModelInstance)
    [
        (model.hogg.σₜ / (model.grid.Δ[1] * model.hogg.ϵ))^2, 
        0.0, 
        (model.economy.σₖ / model.grid.Δ[3])^2
    ]
end

function Q(t, Xᵢ::Point, model::ModelInstance)
    hᵢ = model.grid.h ./ model.grid.Δ
    return dot(hᵢ, bounddrift(t, Xᵢ, model)) + sum(var(model))
end
