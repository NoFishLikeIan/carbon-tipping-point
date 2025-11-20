"Net-zero abatement level."
function ᾱ(t, Xᵢ, model, calibration)
    M = exp(Xᵢ.m) * model.climate.hogg.Mᵖ
    return γ(t, calibration) + δₘ(M, model.climate.decay)
end

"Ratio of abatement to net-zero."
function ε(t, Xᵢ, αᵢ, model, calibration)
    αᵢ / ᾱ(t, Xᵢ, model, calibration)
end
function ε(valuefunction::ValueFunction, model, calibration, G)
    @unpack t, α, H = valuefunction
    
    Tspace, mspace = G.ranges
    E = similar(α)
    @inbounds for (j, m) in enumerate(mspace), (i, T) in enumerate(Tspace)
        x = Point(T, m)
        E[i, j] = ε(t.t, x, α[i, j], model, calibration)
    end

    return E
end

function l(t, Xᵢ, αᵢ, model::M, calibration::Calibration) where {S, M <: UnitIAM{S}}
    @unpack economy, preferences, climate = model

    e = ε(t, Xᵢ, αᵢ, model, calibration)

    abatement = A(t, economy.investments) * β(t, e, economy.abatement)
    damages = d(Xᵢ.T, Xᵢ.m, economy.damages, climate)

    return (preferences.θ - 1) * (damages + abatement)
end