"Emissivity rate implied by abatement `α` at time `t` and carbon concentration `M`"
function ε(t, M, α, model::ModelInstance)
    1f0 - M * (δₘ(M, model.hogg) + γ(t, model.economy, model.calibration) - α) / (Gtonoverppm * Eᵇ(t, model.economy, model.calibration))
end
function ε′(t, M, model::ModelInstance)
    M / (Gtonoverppm * Eᵇ(t, model.economy, model.calibration))
end

"Drift of the state space"
function b(t, Xᵢ::Point, policy::Policy, model::ModelInstance)
    @unpack economy, hogg, albedo, calibration = model

    abatement = A(t, economy) * β(t, ε(t, exp(Xᵢ.m), policy.α, model), economy)
    
    Drift(
        μ(Xᵢ.T, Xᵢ.m, hogg, albedo),
        γ(t, economy, calibration) - policy.α,
        economy.ϱ - economy.δₖᵖ + ϕ(t, policy.χ, economy) - abatement - d(Xᵢ.T, economy, hogg)
    )
end

"Maximum absolute drift of the state space"
function b̄(t, Xᵢ::Point, model::ModelInstance)
    @unpack economy, hogg, albedo, calibration = model
    
    Drift(
        abs(μ(Xᵢ.T, Xᵢ.m, hogg, albedo)),
        γ(t, economy, calibration),
        abs(economy.ϱ - economy.δₖᵖ - d(Xᵢ.T, economy, hogg) + ϕ(t, 0f0, economy))
    )
end

function var(model::ModelInstance)
    [
        (model.hogg.σₜ / (model.grid.Δ[1] * model.hogg.ϵ))^2, 
        0f0, 
        (model.economy.σₖ / model.grid.Δ[3])^2
    ]
end

function Q(t, Xᵢ::Point, model::ModelInstance)
    hᵢ = model.grid.h ./ model.grid.Δ
    drift = dot(hᵢ, Model.b̄(t, Xᵢ, model))

    return drift + sum(var(model))
end

function Δt(t, Xᵢ::Point, model::ModelInstance)
    model.grid.h^2 / Q(t, Xᵢ, model)
end

function p(t, Xᵢ::Point, policy::Policy, model::ModelInstance)
    hᵢ = model.grid.h ./ model.grid.Δ
    bᵢ = b(t, Xᵢ, policy, model)

    b⁺ = max.(bᵢ, 0f0)
    b⁻ = max.(-bᵢ, 0f0)

    σ² = var(model) ./ 2f0

    p = [σ² .+ hᵢ .* b⁺; σ² .+ hᵢ .* b⁻] ./ Q(t, Xᵢ, model)

    p |> TransitionProbability
end