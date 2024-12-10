# TODO: break up these functions
μ(T, m, model::TippingModel) = μ(T, m, model.hogg, model.albedo)
μ(T, m, model::JumpModel) = μ(T, m, model.hogg)
μ(T, m, model::LinearModel) = μ(T, m, model.hogg)

mstable(T, model::TippingModel) = mstable(T, model.hogg, model.albedo)
mstable(T, model::JumpModel) = mstable(T, model.hogg)
mstable(T, model::LinearModel) = mstable(T, model.hogg)

Tstable(m, model::TippingModel) = Tstable(m, model.hogg, model.albedo)
Tstable(m, model::JumpModel) = Tstable(m, model.hogg)
Tstable(m, model::LinearModel) = Tstable(m, model.hogg)

function bterminal(χ::Float64, economy::Economy)
    growth = economy.ϱ - economy.δₖᵖ
    investments = ϕ(economy.τ, χ, economy)

    return growth + investments
end
function bterminal(χ::Float64, model::AbstractModel{LevelDamages, P}) where P <: Preferences
    bterminal(χ, model.economy)
end

function bterminal(_, χ, model::AbstractModel{LevelDamages, P}) where P <: Preferences # This is kept for compatibility, TODO: There is probably a better way to do it.
    bterminal(χ, model)
end
function bterminal(T::Float64, χ::Float64, model::AbstractModel{GrowthDamages, P}) where P <: Preferences
    bterminal(χ, model.economy) - d(T, model.damages, model.hogg)
end

"Emissivity rate implied by abatement `α` at time `t` and carbon concentration `M`"
function ε(t, M, α, model::AbstractModel, calibration::Calibration) 
    α / (δₘ(M, model.hogg) + γ(t, calibration))
end

"Drift of log output y for `t < τ`" # TODO: Combine the drifts
function b(t, Xᵢ::Point, u, model::AbstractModel{GrowthDamages, P}, calibration::Calibration) where P <: Preferences 
    χ, α = u
    εₜ = ε(t, exp(Xᵢ.m), α, model, calibration) 
    Aₜ = A(t, model.economy)

    abatement = Aₜ * β(t, εₜ, model.economy)

    growth = model.economy.ϱ - model.economy.δₖᵖ
    investments = ϕ(t, χ, model.economy)
    damage = d(Xᵢ.T, model.damages, model.hogg)

    return growth + investments - abatement - damage
end
function b(t, Xᵢ::Point, u, model::AbstractModel{LevelDamages, P}, calibration::Calibration) where P <: Preferences
    χ, α = u

    εₜ = ε(t, exp(Xᵢ.m), α, model, calibration)
    Aₜ = A(t, model.economy)

    abatement = Aₜ * β(t, εₜ, model.economy)

    growth = model.economy.ϱ - model.economy.δₖᵖ
    investments = ϕ(t, χ, model.economy)

    return growth + investments - abatement
end

function costbreakdown(t, Xᵢ::Point, u,  model::AbstractModel{GrowthDamages, P}, calibration::Calibration) where P <: Preferences
    χ, α = u
    εₜ = ε(t, exp(Xᵢ.m), α, model, calibration::Calibration)
    Aₜ = A(t, model.economy)

    abatement = β(t, εₜ, model.economy)
    adjcosts = model.economy.κ * β(t, εₜ, model.economy)^2 / 2.
    
    damage = d(Xᵢ.T, model.damages, model.hogg)

    damage, adjcosts, abatement
end

# TODO: Combine the two terminal output function
function terminaloutputfct(Tᵢ, Δt, χ, model::AbstractModel)
    drift = bterminal(Tᵢ, χ, model) - model.preferences.θ * model.economy.σₖ^2 / 2
     
    adj = Δt * (1 - model.preferences.θ) * drift

    return max(1 + adj, 0.)
end

function outputfct(t, Xᵢ::Point, Δt, u, model::AbstractModel, calibration::Calibration)
    drift = b(t, Xᵢ, u, model, calibration) - model.preferences.θ * model.economy.σₖ^2 / 2

    adj = Δt * (1 - model.preferences.θ) * drift

    return max(1 + adj, 0.)
end

function criticaltemperature(model::AbstractModel{GrowthDamages, P}) where P <: Preferences
    maximumgrowth = Tᵢ -> begin
        rate, _ = gss(χ -> bterminal(Tᵢ, χ, model), 0., 1.)
        return rate
    end

    find_zero(maximumgrowth, model.hogg.Tᵖ .+ (0., 15.))
end

"Computes the temperature level for which it is impossible to achieve positive output growth"
function terminalgrid(N, model::AbstractModel{GrowthDamages, P}) where P <: Preferences
    T̄ = criticaltemperature(model)

    Tdomain = (model.hogg.Tᵖ, T̄)
    mdomain = mstable.(Tdomain, model)

    RegularGrid([Tdomain, mdomain], N)
end
function terminalgrid(N, model::AbstractModel{LevelDamages, P}) where P <: Preferences
    Tdomain = model.hogg.Tᵖ .+ (0., 9.)
    mdomain = (mstable(Tdomain[1], model), mstable(Tdomain[2], model))

    RegularGrid([Tdomain, mdomain], N)
end