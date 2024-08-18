# TODO: break up these functions
μ(T, m, model::AbstractTippingModel) = μ(T, m, model.hogg, model.albedo)
μ(T, m, model::AbstractJumpModel) = μ(T, m, model.hogg)
mstable(T, model::AbstractTippingModel) = mstable(T, model.hogg, model.albedo)
mstable(T, model::AbstractJumpModel) = mstable(T, model.hogg)

function bterminal(χ::Float64, economy::Economy)
    growth = economy.ϱ - economy.δₖᵖ
    investments = ϕ(economy.τ, χ, economy)

    return growth + investments
end
function bterminal(χ::Float64, model::AbstractPlannerModel{LevelDamages, P}) where P <: Preferences
    bterminal(χ, model.economy)
end
function bterminal(χ::NTuple{2, Float64}, model::AbstractGameModel{LevelDamages, P}) where P <: Preferences
    bterminal.(χ, model.economy)
end
function bterminal(_, χ, model::AbstractModel{LevelDamages, P}) where P <: Preferences # This is kept for compatibility, TODO: There is probably a better way to do it.
    bterminal(χ, model)
end
function bterminal(T::Float64, χ::Float64, model::AbstractPlannerModel{GrowthDamages, P}) where P <: Preferences
    bterminal(χ, model.economy) - d(T, model.damages, model.hogg)
end
function bterminal(T::Float64, χ::NTuple{2, Float64}, model::AbstractGameModel{GrowthDamages, P}) where P <: Preferences
    bterminal.(χ, model.economy) .- d.(T, model.damages, Ref(model.hogg))
end

"Emissivity rate implied by abatement `α` at time `t` and carbon concentration `M`"
function ε(t, M, α, model::AbstractPlannerModel) 
    α / (δₘ(M, model.hogg) + γ(t, model.calibration))
end
function ε(t, M, α::NTuple{2, Float64}, model::AbstractGameModel)
    α ./ (δₘ(M, model.hogg) .+ γ(t, model.regionalcalibration))
end

"Drift of log output y for `t < τ`" # TODO: Combine the two drifts
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
function b(t, Xᵢ::Point, u::NTuple{2, Policy}, model::AbstractGameModel{GrowthDamages, P}) where P <: Preferences
    αs = @. getproperty(u, :α)
    χs = @. getproperty(u, :χ)

    εₜ = ε(t, exp(Xᵢ.m), αs, model)
    Aₜ = @. A(t, model.economy)

    abatement = @. Aₜ * β(t, εₜ, model.economy)

    growth = @. getproperty(model.economy, :ϱ) - getproperty(model.economy, :δₖᵖ)
    investments = @. ϕ(t, χs, model.economy)
    damage = d.(Xᵢ.T, model.damages, Ref(model.hogg))

    @. growth + investments - abatement - damage
end

# TODO: Combine the two terminal output function
function terminaloutputfct(Tᵢ, Δt, χ, model::AbstractPlannerModel)
    drift = bterminal(Tᵢ, χ, model) - model.preferences.θ * model.economy.σₖ^2 / 2
     
    adj = Δt * (1 - model.preferences.θ) * drift

    return max(1 + adj, 0.)
end
function terminaloutputfct(Tᵢ, Δt, χs, model::AbstractGameModel)
    θs = getproperty.(model.preferences, :θ)

    drift = bterminal(Tᵢ, χs, model) .- θs .* getproperty.(model.economy, :σₖ) .^2 ./ 2
     
    adj = @. Δt * (1 - θs) * drift

    @. max(1 + adj, 0.)
end

function outputfct(t, Xᵢ::Point, Δt, u::Policy, model::AbstractPlannerModel)
    drift = b(t, Xᵢ, u, model) - model.preferences.θ * model.economy.σₖ^2 / 2

    adj = Δt * (1 - model.preferences.θ) * drift

    return max(1 + adj, 0.)
end
function outputfct(t, Xᵢ::Point, Δt, u::NTuple{2, Policy}, model::AbstractGameModel)
    θs = getproperty.(model.preferences, :θ)
    
    drift = b(t, Xᵢ, u, model) .- θs .* getproperty.(model.economy, :σₖ) .^2 ./ 2
     
    adj = @. Δt * (1 - θs) * drift

    @. max(1 + adj, 0.)
end

"Computes the temperature level for which it is impossible to achieve positive output growth"
function constructdefaultgrid(N, model::AbstractModel)
    T̄ = model.hogg.T₀ + (typeof(model.damages) <: LevelDamages ? 8. : 9.)

    Tdomain = (model.hogg.Tᵖ, T̄)
    mdomain = (
        mstable(Tdomain[1], model), 
        mstable(Tdomain[2], model)
    )

    RegularGrid([Tdomain, mdomain], N)
end