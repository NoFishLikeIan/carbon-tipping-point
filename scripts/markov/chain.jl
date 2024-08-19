using Model: AbstractModel, TippingModel, JumpModel
using Grid: Policy

function driftstep(t, idx, F, u::Policy, model::AbstractModel, G)
    L, R = extrema(CartesianIndices(F))
    σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * G.Δ.T))^2
    σₘ² = (model.hogg.σₘ / G.Δ.m)^2

    Xᵢ = G.X[idx]
    dT = μ(Xᵢ.T, Xᵢ.m, model) / (model.hogg.ϵ * G.Δ.T)
    dm = γ(t, model.calibration) - u.α

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

markovstep(t, idx, F, u, model::TippingModel, G) = driftstep(t, idx, F, u, model, G)
function markovstep(t, idx, F, u, model::JumpModel, G)
    Fᵈ, Δt = driftstep(t, idx, F, u, model, G)

    # Update with jump
    R = maximum(CartesianIndices(F))
    Xᵢ = G.X[idx]
    πᵢ = intensity(Xᵢ.T, model.hogg, model.jump)
    qᵢ = increase(Xᵢ.T, model.hogg, model.jump)

    steps = floor(Int, div(qᵢ, G.Δ.T * G.h))
    weight = qᵢ / (G.Δ.T * G.h)

    Fʲ = F[min(idx + steps * I[1], R)] * (1 - weight) + 
            F[min(idx + (steps + 1) * I[1], R)] * weight

    F′ = Fᵈ + πᵢ * Δt * (Fʲ - Fᵈ)

    return F′, Δt
end

function driftstep(t, idx, F, u::Policy, αⱼ, model::AbstractGameModel, G)
    L, R = extrema(CartesianIndices(F))
    σₜ² = (model.hogg.σₜ / (model.hogg.ϵ * G.Δ.T))^2
    σₘ² = (model.hogg.σₘ / G.Δ.m)^2

    Xᵢ = G.X[idx]
    dT = μ(Xᵢ.T, Xᵢ.m, model) / (model.hogg.ϵ * G.Δ.T)
    dm = γ(t, model.regionalcalibration.calibration) - u.α - αⱼ

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

function markovstep(t, idx, F, u, αⱼ, model::TippingGameModel, G)
    driftstep(t, idx, F, u, αⱼ, model, G)
end
function markovstep(t, idx, F, u, αⱼ, model::JumpGameModel, G)
    Fᵈ, Δt = driftstep(t, idx, F, u, αⱼ, model, G)

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

function cost(F′, t, Xᵢ::Point, Δt, u::Policy, model::AbstractModel{GrowthDamages, P}) where P
    δ = outputfct(t, Xᵢ, Δt, u, model)
    g(u.χ, δ * F′, Δt, model.preferences)
end

function cost(F′, t, Xᵢ::Point, Δt, u::Policy, model::AbstractModel{LevelDamages, P}) where P
    δ = outputfct(t, Xᵢ, Δt, u, model)
    damage = d(Xᵢ.T, model.damages, model.hogg)
    g(u.χ * damage, δ * F′, Δt, model.preferences)
end

function isqempty(q)
    all(isempty.(q.minima))
end