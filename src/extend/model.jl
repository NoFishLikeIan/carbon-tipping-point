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

function l(t, Xᵢ, α, model::M, calibration::Calibration) where {S,D<:Damages{S},P<:LogSeparable{S},M<:AbstractModel{S,D,P}}
    @unpack economy, preferences = model
    χ = χopt(t, economy, preferences)
    ε = α / ᾱ(t, Xᵢ, model, calibration)

    gdpgrowth = preferences.ρ * log(χ) + economy.ϱ + ϕ(t, χ, economy) - preferences.θ * economy.σₖ^2 / 2

    netgrowth = gdpgrowth - d(Xᵢ.T, Xᵢ.m, model) - β(t, ε, economy)

    return (1 - preferences.θ) * netgrowth
end

function b(t, Xᵢ::Point, u::Policy, model, calibration)
    @unpack economy = model
    growth = economy.ϱ + ϕ(t, u.χ, economy)

    ε = u.α / ᾱ(t, Xᵢ, model, calibration)
    abatement = A(t, model.economy) * β(t, ε, economy)
    damages = d(Xᵢ.T, Xᵢ.m, model)

    return growth - abatement - damages
end