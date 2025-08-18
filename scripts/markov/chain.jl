function neumannboundary(F, idx, L, R)
    idxT₊ = idx + Idx[1]
    FᵢT₊ = idxT₊.I[1] ≤ R.I[1] ? F[idxT₊] : 2F[idx] - F[idx - Idx[1]]
    
    idxT₋ = idx - Idx[1]
    FᵢT₋ = idxT₋.I[1] ≥ L.I[1] ? F[idxT₋] : 2F[idx] - F[idx + Idx[1]]
    
    idxm₊ = idx + Idx[2]
    Fᵢm₊ = idxm₊.I[2] ≤ R.I[2] ? F[idxm₊] : 2F[idx] - F[idx - Idx[2]]
    
    idxm₋ = idx - Idx[2]
    Fᵢm₋ = idxm₋.I[2] ≥ L.I[2] ? F[idxm₋] : 2F[idx] - F[idx + Idx[2]]
    
    return FᵢT₊, FᵢT₋, Fᵢm₊, Fᵢm₋
end
function constantboundary(F, idx, L, R)
    FᵢT₊ = getindex(F, min(idx + Idx[1], R))
    FᵢT₋ = getindex(F, max(idx - Idx[1], L))
    Fᵢm₊ = getindex(F, min(idx + Idx[2], R))
    Fᵢm₋ = getindex(F, max(idx - Idx[2], L))
    
    return FᵢT₊, FᵢT₋, Fᵢm₊, Fᵢm₋
end
function reflectiveboundary(F, idx, L, R)
    idxT₊ = idx + Idx[1]
    FᵢT₊ = idxT₊.I[1] ≤ R.I[1] ? F[idxT₊] : F[idx - Idx[1]]
    
    idxT₋ = idx - Idx[1]
    FᵢT₋ = idxT₋.I[1] ≥ L.I[1] ? F[idxT₋] : F[idx + Idx[1]]
    
    idxm₊ = idx + Idx[2]
    Fᵢm₊ = idxm₊.I[2] ≤ R.I[2] ? F[idxm₊] : F[idx - Idx[2]]
    
    idxm₋ = idx - Idx[2]
    Fᵢm₋ = idxm₋.I[2] ≥ L.I[2] ? F[idxm₋] : F[idx + Idx[2]]
    
    return FᵢT₊, FᵢT₋, Fᵢm₊, Fᵢm₋
end
function getneighours(F, idx, L, R)
    reflectiveboundary(F, idx, L, R)
end

function timestep(t, Xᵢ::Point, u::Policy, Δtmax, model::AbstractModel, calibration::Calibration, G)
    ΔT, Δm = G.Δ
    σₜ² = ( model.hogg.σₜ / (model.hogg.ϵ * ΔT) )^2
    σₘ² = ( model.hogg.σₘ / Δm )^2

    dT = μ(Xᵢ.T, Xᵢ.m, model) / ( model.hogg.ϵ * ΔT )
    dm = ( γ(t, calibration) * (1 - u.ε) - δₘ(Xᵢ.m, model.hogg) * u.ε )  / Δm

    Q = σₘ² + σₜ² + G.h * ( abs(dT) + abs(dm) )

    return min(G.h^2 / Q, Δtmax)
end

function driftstep(t, idx, F, u::Policy, Δtmax, model::AbstractModel, calibration::Calibration, G)
    ΔT, Δm = G.Δ

    L, R = extrema(G)

    σₜ² = ( model.hogg.σₜ / (model.hogg.ϵ * ΔT) )^2
    σₘ² = ( model.hogg.σₘ / Δm )^2

    Xᵢ = G.X[idx]
    dT = μ(Xᵢ.T, Xᵢ.m, model) / (model.hogg.ϵ * ΔT)
    dm = ( γ(t, calibration) * (1 - u.ε) - δₘ(Xᵢ.m, model.hogg) * u.ε )  / Δm

    FᵢT₊, FᵢT₋, Fᵢm₊, Fᵢm₋ = getneighours(F, idx, L, R)

    Q = σₘ² + σₜ² + G.h * (abs(dT) + abs(dm))

    dTF = G.h * abs(dT) * ifelse(dT > 0, FᵢT₊, FᵢT₋) + σₜ² * (FᵢT₊ + FᵢT₋) / 2
    dmF = G.h * abs(dm) * ifelse(dm > 0, Fᵢm₊, Fᵢm₋) + σₘ² * (Fᵢm₊ + Fᵢm₋) / 2

    F′ = (dTF + dmF) / Q
    Δt = min(G.h^2 / Q, Δtmax)

    return F′, Δt
end

markovstep(t, idx, F, u::Policy, Δtmax, model::Union{LinearModel, TippingModel}, calibration::Calibration, G) = driftstep(t, idx, F, u, Δtmax, model, calibration, G)
function markovstep(t, idx, F, u::Policy, Δtmax, model::JumpModel, calibration::Calibration, G)
    throw("markovstep not implemented for JumpModel!")
    Fᵈ, Δt = driftstep(t, idx, F, α, Δtmax, model, calibration, G)

    # Update with jump
    R = maximum(CartesianIndices(F))
    Xᵢ = G.X[idx]
    πᵢ = intensity(Xᵢ.T, model.hogg, model.jump)
    qᵢ = increase(Xᵢ.T, model.hogg, model.jump)

    steps = floor(Int, div(qᵢ, G.Δ.T * G.h))
    weight = qᵢ / (G.Δ.T * G.h)

    Fʲ = F[min(idx + steps * Idx[1], R)] * (1 - weight) + F[min(idx + (steps + 1) * Idx[1], R)] * weight

    F′ = Fᵈ + πᵢ * Δt * (Fʲ - Fᵈ)

    return F′, Δt
end

function logcost(logF′, Δt, t, Xᵢ::Point, u, model::AbstractModel{T, D}) where {T, D <: GrowthDamages{T}}
    logδF′ = logoutputfct(t, Δt, Xᵢ, u, model) + logF′
    return logg(u.χ, logδF′, Δt, model.preferences)
end
function cost(F′, t, Δt, Xᵢ::Point, u::Policy, model::AbstractModel{T, D}) where {T, D <: GrowthDamages{T}}
    δ = outputfct(t, Δt, Xᵢ, u, model)
    return g(u.χ, Δt, δ * F′, model.preferences)
end

function Base.isempty(q::ZigZagBoomerang.PartialQueue)
    all((isempty(m) for m in q.minima))
end