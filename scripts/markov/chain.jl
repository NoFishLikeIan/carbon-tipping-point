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

function getneighours(F, idx, L, R)
    constantboundary(F, idx, L, R)
end

function timestep(t, Xᵢ::Point, α, model::AbstractModel, calibration::Calibration, G)
    ΔT, Δm = G.Δ
    σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * ΔT))^2
    σₘ² = (model.hogg.σₘ / Δm)^2

    dT = μ(Xᵢ.T, Xᵢ.m, model) / (model.hogg.ϵ * ΔT)
    dm = (γ(t, calibration) - α) / Δm

    Q = σₘ² + σₜ² + G.h * (abs(dT) + abs(dm))

    return G.h^2 / Q
end

function driftstep(t, idx, F, α, model::AbstractModel, calibration::Calibration, G)
    ΔT, Δm = G.Δ

    L, R = extrema(G)

    σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * ΔT))^2
    σₘ² = (model.hogg.σₘ / Δm)^2

    Xᵢ = G.X[idx]
    dT = μ(Xᵢ.T, Xᵢ.m, model) / (model.hogg.ϵ * ΔT)
    dm = (γ(t, calibration) - α) / Δm

    FᵢT₊, FᵢT₋, Fᵢm₊, Fᵢm₋ = getneighours(F, idx, L, R)

    Q = σₘ² + σₜ² + G.h * (abs(dT) + abs(dm))

    dTF = G.h * abs(dT) * ifelse(dT > 0, FᵢT₊, FᵢT₋) + σₜ² * (FᵢT₊ + FᵢT₋) / 2
    dmF = G.h * abs(dm) * ifelse(dm > 0, Fᵢm₊, Fᵢm₋) + σₘ² * (Fᵢm₊ + Fᵢm₋) / 2

    F′ = (dTF + dmF) / Q
    Δt = G.h^2 / Q

    return F′, Δt
end

markovstep(t, idx, F, α, model::Union{LinearModel, TippingModel}, calibration::Calibration, G) = driftstep(t, idx, F, α, model, calibration, G)
function markovstep(t, idx, F, α, model::JumpModel, calibration::Calibration, G)
    Fᵈ, Δt = driftstep(t, idx, F, α, model, calibration, G)

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

function logcost(F′, t, Xᵢ::Point, Δt, u, model::AbstractModel{T, D}, calibration::Calibration) where {T, D <: GrowthDamages{T}}
    δ = outputfct(t, Xᵢ, Δt, u, model, calibration)
    return logg(u.χ, δ * F′, Δt, model.preferences)
end
function cost(F′, t, Xᵢ::Point, Δt, u::Policy, model::AbstractModel{T, D}, calibration::Calibration) where {T, D <: GrowthDamages{T}}
    δ = outputfct(t, Xᵢ, Δt, u, model, calibration)
    return g(u.χ, δ * F′, Δt, model.preferences)
end

function Base.isempty(q::ZigZagBoomerang.PartialQueue)
    all((isempty(m) for m in q.minima))
end