function Model.d(T, m, model::TippingModel)
    Model.d(T, m, model.damages, model.hogg, model.feedback)
end
function Model.d(T, m, model::LinearModel)
    Model.d(T, m, model.damages, model.hogg)
end

"Drift of log output y for `t ≥ τ`"
function bterminal(τ, Xᵢ::Point{T}, χ, model::M) where {T, D <: GrowthDamages{T}, M <: AbstractModel{T, D}}
    growth = model.economy.ϱ - model.economy.δₖᵖ
    investments = ϕ(τ, χ, model.economy)
    damage = d(Xᵢ.T, Xᵢ.m, model)

    return growth + investments - damage
end
function bterminal(τ, χ, model::M) where {T, D <:  LevelDamages{T}, M <: AbstractModel{T, D}}
    growth = model.economy.ϱ - model.economy.δₖᵖ
    investments = ϕ(τ, χ, model.economy)

    return growth + investments
end
function bterminal(τ, _, χ, model::M) where {T, D <:  LevelDamages{T}, M <: AbstractModel{T, D}}
    bterminal(τ, χ, model) # For compatibility
end

"Emissivity rate implied by abatement `α` at time `t` and carbon concentration `M`"
function ε(t, M, α, model, calibration::Calibration) 
    α / (δₘ(M, model.hogg) + γ(t, calibration))
end
function ε(t, M, α, model, regionalcalibration::RegionalCalibration, p)
    α / (δₘ(M, model.hogg) + getindex(γ(t, regionalcalibration), p))
end

"Drift of log output y for `t < τ`"
function b(t, Xᵢ::Point{T}, emissivity, investments, model::M) where { T, D <: GrowthDamages{T}, P <: Preferences, M <: AbstractModel{T, D, P} } 
    Aₜ = A(t, model.economy)

    abatement = Aₜ * β(t, emissivity, model.economy)

    growth = model.economy.ϱ - model.economy.δₖᵖ
    damage = d(Xᵢ.T, Xᵢ.m, model)

    return growth + investments - abatement - damage
end
function b(t, _, emissivity, investments, model::M) where { T, D <: LevelDamages{T}, P <: Preferences{T}, M <: AbstractModel{T, D, P} }
    Aₜ = A(t, model.economy)
    abatement = Aₜ * β(t, emissivity, model.economy)

    return growth + investments - abatement
end
function b(t, Xᵢ::Point, u::Policy, model::M, calibration::Calibration) where {T, D <: GrowthDamages{T}, M <: AbstractModel{T, D}}
    εₜ = ε(t, Xᵢ.M, u.α, model, calibration) 
    investments = ϕ(t, u.χ, model.economy)
    
    return b(t, Xᵢ, εₜ, investments, model)
end
function b(t, Xᵢ::Point{T}, u, model::M, calibration::RegionalCalibration, p) where {T, D <: GrowthDamages{T}, M <: AbstractModel{T, D}}
    χ, α = u
    εₜ = ε(t, Xᵢ.M, α, model, calibration, p) 
    investments = ϕ(t, χ, model.economy)
    
    return b(t, Xᵢ, εₜ, investments, model)
end

# TODO: Update the cost breakdown for the game model
function costbreakdown(t, Xᵢ::Point{T}, u,  model::M, calibration) where {T, D <: GrowthDamages{T}, P <: Preferences, M <: AbstractModel{T, D, P}}
    χ, α = u
    εₜ = ε(t, Xᵢ.M, α, model, calibration)
    Aₜ = A(t, model.economy)

    abatement = β(t, εₜ, model.economy)
    adjcosts = model.economy.κ * β(t, εₜ, model.economy)^2 / 2.
    
    damage = d(Xᵢ.T, model.damages, model.hogg)

    damage, adjcosts, abatement
end

"δ-factor for output at time `t ≥ τ`"
function terminaloutputfct(τ, Xᵢ::Point{T}, Δt, χ, model::M) where { T, M <: AbstractModel{T}}
    drift = bterminal(τ, Xᵢ, χ, model) - model.preferences.θ * model.economy.σₖ^2 / 2
     
    adj = Δt * (1 - model.preferences.θ) * drift

    return max(1 + adj, 0)
end

"δ-factor for output at time `t < τ`"
function outputfct(t, Xᵢ::Point, Δt, u, model, calibration::Calibration)
    drift = b(t, Xᵢ, u, model, calibration) - model.preferences.θ * model.economy.σₖ^2 / 2

    adj = Δt * (1 - model.preferences.θ) * drift

    return max(1 + adj, 0)
end
function outputfct(t, Xᵢ::Point, Δt, u, model, calibration::RegionalCalibration, p)
    drift = b(t, Xᵢ, u, model, calibration, p) - model.preferences.θ * model.economy.σₖ^2 / 2

    adj = Δt * (1 - model.preferences.θ) * drift

    return max(1 + adj, 0)
end


function Grid.RegularGrid(domains::NTuple{2, Domain{R}}, N::Int, hogg::Hogg) where R
    if N > Grid.maxN @warn "h < ϵ: ensure N ≤ $Grid.maxN" end

    h = 1 / (N - 1)
    Tdomain, mdomain = domains
    ΔT = Tdomain[2] - Tdomain[1]
    Δm = mdomain[2] - mdomain[1]
    
    Tspace = range(Tdomain[1], Tdomain[2]; length = N)
    mspace = range(mdomain[1], mdomain[2]; length = N)

    X = [ Point(T, m, hogg.Mᵖ * exp(m)) for T in Tspace, m in mspace]
    
    Grid.RegularGrid{N, R}(h, X, (ΔT, Δm), domains)
end
function Grid.RegularGrid(domains::AbstractVector{Domain}, h::T) where T
    N = floor(Int, 1 / h) + 1
    return Grid.RegularGrid(domains, N)
end

function Grid.interpolateovergrid(state::DPState, fromgrid::RegularGrid, togrid::RegularGrid)
    Fₜ = interpolateovergrid(state.valuefunction.Fₜ, fromgrid, togrid)
    Fₜ₊ₕ = copy(Fₜ)
    errors = interpolateovergrid(state.valuefunction.error, fromgrid, togrid)
    valuefunction = ValueFunction(Fₜ, Fₜ₊ₕ, errors)

    policy = interpolateovergrid(state.policystate.policy, fromgrid, togrid)
	foc = interpolateovergrid(state.policystate.foc, fromgrid, togrid)
    policystate = PolicyState(policy, foc)

    t = interpolateovergrid(state.timestate.t, fromgrid, togrid)

    return DPState(valuefunction, policystate, Time(state.timestate.τ, t))
end