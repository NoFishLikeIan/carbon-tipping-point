function bterminal(Xᵢ::Point, χ, model::ModelInstance)
    @unpack economy, hogg = model
    ϕ(economy.τ, χ, economy) - economy.δₖᵖ - d(Xᵢ.T, economy, hogg)
end

function b(t, Xᵢ::Point, policy::Policy, model::ModelInstance)
    @unpack economy, hogg = model

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

"Drift on the unit cube"
function drift(t, Xᵢ::Point, policy::Policy, model::ModelInstance)
    @unpack economy, hogg, albedo, calibration = model
    
    Drift(
        μ(Xᵢ.T, Xᵢ.m, hogg, albedo) / (hogg.ϵ * model.grid.Δ[1]),
        (γ(t, economy, calibration) - policy.α) / model.grid.Δ[2],
        b(t, Xᵢ, policy, model) / model.grid.Δ[3]
    )
end

"Maximum absolute drift on the unit cube"
function bounddrift(t, Xᵢ::Point, model::ModelInstance)
    @unpack economy, hogg, albedo, calibration = model
    
    Drift(
        abs(μ(Xᵢ.T, Xᵢ.m, hogg, albedo) / (hogg.ϵ * model.grid.Δ[1])),
        γ(t, economy, calibration) / model.grid.Δ[2],
        abs(b(t, Xᵢ, Policy(0f0, 0f0), model)) / model.grid.Δ[3]
    )
end

function terminaldrift(Xᵢ::Point, χ, model::ModelInstance)
    TerminalDrift(
        μ(Xᵢ.T, Xᵢ.m, model.hogg, model.albedo) / (model.hogg.ϵ * model.grid.Δ[1]),
        bterminal(Xᵢ, χ, model) / model.grid.Δ[3]
    )
end

function boundterminaldrift(Xᵢ::Point, model::ModelInstance)
    @unpack economy, hogg, albedo, calibration = model
    
    TerminalDrift(
        abs(μ(Xᵢ.T, Xᵢ.m, hogg, albedo) / (model.hogg.ϵ * model.grid.Δ[1])),
        abs(bterminal(Xᵢ, 0f0, model)) / model.grid.Δ[3]
    )
end

"Computes the normalised square variances of the model"
function σ̃²(model::ModelInstance)
    (model.hogg.σₜ / (model.grid.Δ[1] * model.hogg.ϵ))^2, (model.economy.σₖ / model.grid.Δ[3])^2
end
