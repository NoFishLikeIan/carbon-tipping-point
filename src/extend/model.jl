function Model.d(T, m, model::TippingModel)
    Model.d(T, m, model.damages, model.hogg, model.feedback)
end
function Model.d(T, m, model::LinearModel)
    Model.d(T, m, model.damages, model.hogg)
end

function ᾱ(t, Xᵢ, model, calibration)
    M = exp(Xᵢ.m) * model.hogg.Mᵖ
    return γ(t, calibration) + δₘ(M, model.hogg)
end

function ε(t, Xᵢ, αᵢ, model, calibration)
    αᵢ / ᾱ(t, Xᵢ, model, calibration)
end

function ε(valuefunction::ValueFunction, model, calibration)
    @unpack t, α, H = valuefunction

    E = similar(α)
    @inbounds for idx in CartesianIndices(G)
        E[idx] = ε(t.t, G[idx], α[idx], model, calibration)
    end

    return E
end

function l(t, Xᵢ, αᵢ, model::M, calibration::Calibration) where {S, M <: UnitElasticityModel{S}}
    @unpack economy, preferences = model
    χ = χopt(t, economy, preferences)
    e = ε(t, Xᵢ, αᵢ, model, calibration)

    gdpgrowth = preferences.ρ * log(χ) + economy.ϱ + ϕ(t, χ, economy) - preferences.θ * economy.σₖ^2 / 2

    netgrowth = gdpgrowth - d(Xᵢ.T, Xᵢ.m, model) - A(t, economy) * β(t, e, economy)

    return (1 - preferences.θ) * netgrowth
end

function b(t, Xᵢ::Point, u::Policy, model, calibration)
    @unpack economy = model
    growth = economy.ϱ + ϕ(t, u.χ, economy)
    abatement = A(t, model.economy) * β(t, ε(t, Xᵢ, u.α, model, calibration), economy)
    damages = d(Xᵢ.T, Xᵢ.m, model)

    return growth - abatement - damages
end