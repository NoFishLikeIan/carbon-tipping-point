function ᾱ(t, Xᵢ, model, calibration)
    M = exp(Xᵢ.m) * model.climate.hogg.Mᵖ
    return γ(t, calibration) + δₘ(M, model.climate.decay)
end

function ε(t, Xᵢ, αᵢ, model, calibration)
    αᵢ / ᾱ(t, Xᵢ, model, calibration)
end
function ε(valuefunction::ValueFunction, model, calibration, G)
    @unpack t, α, H = valuefunction

    E = similar(α)
    @inbounds for idx in CartesianIndices(G)
        E[idx] = ε(t.t, G[idx], α[idx], model, calibration)
    end

    return E
end

function l(t, Xᵢ, αᵢ, model::M, calibration::Calibration) where {S, M <: UnitIAM{S}}
    @unpack economy, preferences, climate = model

    χ = χopt(t, economy, preferences)
    e = ε(t, Xᵢ, αᵢ, model, calibration)

    growth = ϕ(t, χ, economy.investments) + economy.investments.ϱ
    abatement = A(t, economy.investments) * β(t, e, economy.abatement)
    damages = d(Xᵢ.T, Xᵢ.m, economy.damages, climate)
    consumption = preferences.ρ * log(χ) - preferences.θ * economy.investments.σₖ^2 / 2

    return (1 - preferences.θ) * (consumption + growth - damages - abatement)
end