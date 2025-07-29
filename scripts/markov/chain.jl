function timestep(t, Xᵢ::Point, α, model::AbstractModel, calibration::Calibration, G)
    σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * G.Δ.T))^2
    σₘ² = (model.hogg.σₘ / G.Δ.m)^2

    dT = μ(Xᵢ.T, Xᵢ.m, model) / (model.hogg.ϵ * G.Δ.T)
    dm = (γ(t, calibration) - α) / G.Δ.m

    Q = σₘ² + σₜ² + G.h * (abs(dT) + abs(dm))

    return G.h^2 / Q
end

function driftstep(t, idx, F, α, model::AbstractModel, calibration::Calibration, G)
    L, R = extrema(CartesianIndices(F))

    σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * G.Δ.T))^2
    σₘ² = (model.hogg.σₘ / G.Δ.m)^2

    Xᵢ = G.X[idx]
    dT = μ(Xᵢ.T, Xᵢ.m, model) / (model.hogg.ϵ * G.Δ.T)
    dm = (γ(t, calibration) - α) / G.Δ.m

    # -- Temperature
    FᵢT₊ = F[min(idx + I[1], R)]
    FᵢT₋ = F[max(idx - I[1], L)]
    # -- Carbon concentration
    Fᵢm₊ = F[min(idx + I[2], R)]
    Fᵢm₋ = F[max(idx - I[2], L)]

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

    Fʲ = F[min(idx + steps * I[1], R)] * (1 - weight) + F[min(idx + (steps + 1) * I[1], R)] * weight

    F′ = Fᵈ + πᵢ * Δt * (Fʲ - Fᵈ)

    return F′, Δt
end

function logcost(F′, t, Xᵢ::Point, Δt, u, model::AbstractModel, calibration::RegionalCalibration, p)
    δ = outputfct(t, Xᵢ, Δt, u, model, calibration, p)
    logcost(F′, δ, Xᵢ, Δt, u, model)
end
function logcost(F′, t, Xᵢ::Point, Δt, u, model::AbstractModel, calibration::Calibration)
    δ = outputfct(t, Xᵢ, Δt, u, model, calibration)
    logcost(F′, δ, Xᵢ, Δt, u, model)
end
function logcost(F′, δ, Xᵢ::Point, Δt, u, model::AbstractModel{GrowthDamages})
    logg(u[1], δ * F′, Δt, model.preferences)
end
function logcost(F′, δ, Xᵢ::Point, Δt, u, model::AbstractModel{LevelDamages})
    damage = d(Xᵢ.T, model.damages, model.hogg)
    logg(u[1] * damage, δ * F′, Δt, model.preferences)
end

function cost(F′, t, Xᵢ::Point, Δt, u, model::AbstractModel, calibration::RegionalCalibration, p)
    δ = outputfct(t, Xᵢ, Δt, u, model, calibration, p)
    cost(F′, δ, Xᵢ, Δt, u, model)
end
function cost(F′, t, Xᵢ::Point, Δt, u, model::AbstractModel, calibration::Calibration)
    δ = outputfct(t, Xᵢ, Δt, u, model, calibration)
    cost(F′, δ, Xᵢ, Δt, u, model)
end
function cost(F′, δ, _, Δt, u, model::AbstractModel{GrowthDamages, P}) where P
    g(u[1], δ * F′, Δt, model.preferences)
end
function cost(F′, δ, Xᵢ::Point, Δt, u, model::AbstractModel{LevelDamages, P}) where P
    damage = d(Xᵢ.T, model.damages, model.hogg)
    g(u[1] * damage, δ * F′, Δt, model.preferences)
end


function Base.isempty(q::ZigZagBoomerang.PartialQueue)
    all((isempty(m) for m in q.minima))
end