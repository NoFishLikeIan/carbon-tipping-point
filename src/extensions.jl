function Model.d(Xᵢ::Point{T}, model::TippingModel{T, GrowthDamages}) where T
    Model.d(Xᵢ.T, Xᵢ.m, model.damages, model.hogg, model.feedback)
end
function Model.d(Xᵢ::Point{T}, model::LinearModel{T, GrowthDamages}) where T
    Model.d(Xᵢ.T, Xᵢ.m, model.damages, model.hogg)
end
function Model.d(Xᵢ::Point{T}, model::AbstractModel{T, LevelDamages}) where T
    Model.d(Xᵢ.T, model.damages, model.hogg)
end

"Drift of log output y for `t ≥ τ`"
function bterminal(Xᵢ::Point{T}, χ, model::AbstractModel{T, GrowthDamages}) where T
    growth = model.economy.ϱ - model.economy.δₖᵖ
    investments = ϕ(model.economy.τ, χ, model.economy)
    damage = d(Xᵢ, model)

    return growth + investments - damage
end
function bterminal(χ, model::AbstractModel{T, LevelDamages}) where T
    growth = model.economy.ϱ - model.economy.δₖᵖ
    investments = ϕ(model.economy.τ, χ, model.economy)

    return growth + investments
end
bterminal(_, χ, model::AbstractModel{T, LevelDamages}) where T = bterminal(χ, model) # For compatibility

"Emissivity rate implied by abatement `α` at time `t` and carbon concentration `M`"
function ε(t, M, α, model::AbstractModel, calibration::Calibration) 
    α / (δₘ(M, model.hogg) + γ(t, calibration))
end
function ε(t, M, α, model::AbstractModel, regionalcalibration::RegionalCalibration, p)
    α / (δₘ(M, model.hogg) + getindex(γ(t, regionalcalibration), p))
end

"Drift of log output y for `t < τ`"
function b(t, Xᵢ::Point{T}, emissivity, investments, model::AbstractModel{T, GrowthDamages, P}) where { T, P <: Preferences } 
    Aₜ = A(t, model.economy)

    abatement = Aₜ * β(t, emissivity, model.economy)

    growth = model.economy.ϱ - model.economy.δₖᵖ
    damage = d(Xᵢ, model)

    return growth + investments - abatement - damage
end
function b(t, Xᵢ::Point{T}, emissivity, investments, model::AbstractModel{T, LevelDamages, P}) where { T, P <: Preferences }
    Aₜ = A(t, model.economy)
    abatement = Aₜ * β(t, emissivity, model.economy)

    return growth + investments - abatement
end
function b(t, Xᵢ::Point{T}, u, model::AbstractModel{T, GrowthDamages}, calibration::Calibration) where T
    χ, α = u
    M = exp(Xᵢ.m) * model.hogg.Mᵖ
    εₜ = ε(t, M, α, model, calibration) 
    investments = ϕ(t, χ, model.economy)
    
    return b(t, Xᵢ, εₜ, investments, model)
end
function b(t, Xᵢ::Point{T}, u, model::AbstractModel{T, GrowthDamages}, calibration::RegionalCalibration, p) where T
    χ, α = u
    M = exp(Xᵢ.m) * model.hogg.Mᵖ
    εₜ = ε(t, M, α, model, calibration, p) 
    investments = ϕ(t, χ, model.economy)
    
    return b(t, Xᵢ, εₜ, investments, model)
end

# TODO: Update the cost breakdown for the game model
function costbreakdown(t, Xᵢ::Point{T}, u,  model::AbstractModel{T, GrowthDamages, P}, calibration) where {T, P <: Preferences}
    χ, α = u
    M = exp(Xᵢ.m) * model.hogg.Mᵖ
    εₜ = ε(t, M, α, model, calibration)
    Aₜ = A(t, model.economy)

    abatement = β(t, εₜ, model.economy)
    adjcosts = model.economy.κ * β(t, εₜ, model.economy)^2 / 2.
    
    damage = d(Xᵢ.T, model.damages, model.hogg)

    damage, adjcosts, abatement
end

"δ-factor for output at time `t ≥ τ`"
function terminaloutputfct(Xᵢ::Point{T}, Δt, χ, model::AbstractModel{T}) where T
    drift = bterminal(Xᵢ, χ, model) - model.preferences.θ * model.economy.σₖ^2 / 2
     
    adj = Δt * (1 - model.preferences.θ) * drift

    return max(1 + adj, zero(T))
end

"δ-factor for output at time `t < τ`"
function outputfct(t, Xᵢ::Point{T}, Δt, u, model::AbstractModel{T}, calibration::Calibration) where T
    drift = b(t, Xᵢ, u, model, calibration) - model.preferences.θ * model.economy.σₖ^2 / 2

    adj = Δt * (1 - model.preferences.θ) * drift

    return max(1 + adj, zero(T))
end
function outputfct(t, Xᵢ::Point{T}, Δt, u, model::AbstractModel{T}, calibration::RegionalCalibration, p) where T
    drift = b(t, Xᵢ, u, model, calibration, p) - model.preferences.θ * model.economy.σₖ^2 / 2

    adj = Δt * (1 - model.preferences.θ) * drift

    return max(1 + adj, zero(T))
end