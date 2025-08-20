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

"Drift of log output y for `t < τ`"
function b(t, Xᵢ::Point, u::Policy, model::M) where { T, D <: GrowthDamages{T}, P <: Preferences, M <: AbstractModel{T, D, P} } 
    abatement = A(t, model.economy) * β(t, u.ε, model.economy)
    investments = ϕ(t, u.χ, model.economy)
    growth = model.economy.ϱ - model.economy.δₖᵖ
    damages = d(Xᵢ.T, Xᵢ.m, model)

    return growth + investments - abatement - damages
end
function b(t, u::Policy, model::M) where { T, D <: LevelDamages{T}, P <: Preferences{T}, M <: AbstractModel{T, D, P} }
    abatement = A(t, model.economy) * β(t, u.ε, model.economy)
    investments = ϕ(t, u.χ, model.economy)
    growth = model.economy.ϱ - model.economy.δₖᵖ

    return growth + investments - abatement
end
b(t, _, u::Policy, model::M) where { T, D <: LevelDamages{T}, P <: Preferences{T}, M <: AbstractModel{T, D, P} } = b(t, u, model) # Consistency of signature

function costbreakdown(t, Xᵢ::Point, u::Policy, model::M, calibration) where {T, D <: GrowthDamages, P <: Preferences, M <: AbstractModel{T, D, P}}
    abatement = β(t, u.ε, model.economy)
    adjustments = model.economy.κ * abatement^2 / 2
    damages = d(Xᵢ.T, Xᵢ.m, model.damages, model.hogg)

    return abatement, adjustments, damages
end

"δ-factor for output at time `t ≥ τ`"
function terminaloutputfct(τ, Δt, Xᵢ::Point{T}, χ, model::M) where { T, M <: AbstractModel{T}}
    drift = bterminal(τ, Xᵢ, χ, model) - model.preferences.θ * model.economy.σₖ^2 / 2
     
    adj = Δt * (1 - model.preferences.θ) * drift

    return max(1 + adj, 0)
end
function logterminaloutputfct(τ, Δt, Xᵢ::Point{T}, χ, model::M) where { T, M <: AbstractModel{T}}
    drift = bterminal(τ, Xᵢ, χ, model) - model.preferences.θ * model.economy.σₖ^2 / 2
     
    adj = Δt * (1 - model.preferences.θ) * drift

    return log1p(max(adj, -1))
end


"δ-factor for output at time `t < τ`"
function outputfct(t, Δt, Xᵢ::Point, u::Policy, model)
    drift = b(t, Xᵢ, u, model) - model.preferences.θ * model.economy.σₖ^2 / 2

    adj = Δt * (1 - model.preferences.θ) * drift

    return max(1 + adj, 0)
end
function logoutputfct(t, Δt, Xᵢ::Point, u::Policy, model)
    drift = b(t, Xᵢ, u, model) - model.preferences.θ * model.economy.σₖ^2 / 2
    
    adj = Δt * (1 - model.preferences.θ) * drift
    
    return log1p(max(adj, -1))
end

function constructgrid(domains::NTuple{2, Domain{R}}, N::Int, hogg::Hogg) where R
    if N > Grid.maxN @warn "h < ϵ: ensure N ≤ $Grid.maxN" end

    h = 1 / (N - 1)
    Tdomain, mdomain = domains
    ΔT = Tdomain[2] - Tdomain[1]
    Δm = mdomain[2] - mdomain[1]
    
    Tspace = range(Tdomain[1], Tdomain[2]; length = N)
    mspace = range(mdomain[1], mdomain[2]; length = N)

    X = [ Point(T, m, hogg.Mᵖ * exp(m)) for T in Tspace, m in mspace]
    
    return Grid.RegularGrid{N, R}(h, X, (ΔT, Δm), domains)
end
function constructgrid(domains::AbstractVector{Domain}, h::T) where T
    N = floor(Int, 1 / h) + 1
    return RegularGrid(domains, N)
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

function shrink(domain::Domain, factor)
    l, r = domain
    cut = (r - l) * (1 - factor) / 2

    return (l + cut, r - cut)
end

function shrink(G::RegularGrid{N}, factor, hogg::Hogg) where N
    domains = shrink.(G.domains, factor)

    return constructgrid(domains, N, hogg)
end