function Model.d(T, m, model::TippingModel)
    Model.d(T, m, model.damages, model.hogg, model.feedback)
end
function Model.d(T, m, model::LinearModel)
    Model.d(T, m, model.damages, model.hogg)
end

"Drift of log output y for `t ≥ τ`"
function bterminal(τ, Xᵢ::Point{T}, χ, model::AbstractModel{T, D}) where {T, D <: GrowthDamages{T}}
    growth = model.economy.ϱ - model.economy.δₖᵖ
    investments = ϕ(τ, χ, model.economy)
    damage = d(Xᵢ.T, Xᵢ.m, model)

    return growth + investments - damage
end
function bterminal(τ, χ, model::AbstractModel{T, D}) where {T, D <:  LevelDamages{T}}
    growth = model.economy.ϱ - model.economy.δₖᵖ
    investments = ϕ(τ, χ, model.economy)

    return growth + investments
end
function bterminal(τ, _, χ, model::AbstractModel{T, D}) where {T, D <:  LevelDamages{T}}
    bterminal(τ, χ, model) # For compatibility
end

"Emissivity rate implied by abatement `α` at time `t` and carbon concentration `M`"
function ε(t, M, α, model::AbstractModel, calibration::Calibration) 
    α / (δₘ(M, model.hogg) + γ(t, calibration))
end
function ε(t, M, α, model::AbstractModel, regionalcalibration::RegionalCalibration, p)
    α / (δₘ(M, model.hogg) + getindex(γ(t, regionalcalibration), p))
end

"Drift of log output y for `t < τ`"
function b(t, Xᵢ::Point{T}, emissivity, investments, model::AbstractModel{T, D, P}) where { T, D <: GrowthDamages{T}, P <: Preferences } 
    Aₜ = A(t, model.economy)

    abatement = Aₜ * β(t, emissivity, model.economy)

    growth = model.economy.ϱ - model.economy.δₖᵖ
    damage = d(Xᵢ.T, Xᵢ.m, model)

    return growth + investments - abatement - damage
end
function b(t, Xᵢ::Point{T}, emissivity, investments, model::AbstractModel{T, D, P}) where { T, D <: LevelDamages{T}, P <: Preferences }
    Aₜ = A(t, model.economy)
    abatement = Aₜ * β(t, emissivity, model.economy)

    return growth + investments - abatement
end
function b(t, Xᵢ::Point, u::Policy, model::AbstractModel{T, D}, calibration::Calibration) where {T, D <: GrowthDamages{T}}
    εₜ = ε(t, Xᵢ.M, u.α, model, calibration) 
    investments = ϕ(t, u.χ, model.economy)
    
    return b(t, Xᵢ, εₜ, investments, model)
end
function b(t, Xᵢ::Point{T}, u, model::AbstractModel{T, D}, calibration::RegionalCalibration, p) where {T, D <: GrowthDamages{T}}
    χ, α = u
    εₜ = ε(t, Xᵢ.M, α, model, calibration, p) 
    investments = ϕ(t, χ, model.economy)
    
    return b(t, Xᵢ, εₜ, investments, model)
end

# TODO: Update the cost breakdown for the game model
function costbreakdown(t, Xᵢ::Point{T}, u,  model::AbstractModel{T, D, P}, calibration) where {T, D <: GrowthDamages{T}, P <: Preferences}
    χ, α = u
    εₜ = ε(t, Xᵢ.M, α, model, calibration)
    Aₜ = A(t, model.economy)

    abatement = β(t, εₜ, model.economy)
    adjcosts = model.economy.κ * β(t, εₜ, model.economy)^2 / 2.
    
    damage = d(Xᵢ.T, model.damages, model.hogg)

    damage, adjcosts, abatement
end

"δ-factor for output at time `t ≥ τ`"
function terminaloutputfct(τ, Xᵢ::Point{T}, Δt, χ, model::AbstractModel{T}) where T
    drift = bterminal(τ, Xᵢ, χ, model) - model.preferences.θ * model.economy.σₖ^2 / 2
     
    adj = Δt * (1 - model.preferences.θ) * drift

    return max(1 + adj, 0)
end

"δ-factor for output at time `t < τ`"
function outputfct(t, Xᵢ::Point, Δt, u, model::AbstractModel, calibration::Calibration)
    drift = b(t, Xᵢ, u, model, calibration) - model.preferences.θ * model.economy.σₖ^2 / 2

    adj = Δt * (1 - model.preferences.θ) * drift

    return max(1 + adj, 0)
end
function outputfct(t, Xᵢ::Point, Δt, u, model::AbstractModel, calibration::RegionalCalibration, p)
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
    return RegularGrid(domains, N)
end