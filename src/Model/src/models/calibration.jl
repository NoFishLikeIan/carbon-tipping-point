abstract type Calibration{S <: Real} end

Base.@kwdef struct ConstantCalibration{S} <: Calibration{S}
   γ₀::S = 0.022
end

struct DynamicCalibration{S} <: Calibration{S}
    calibrationspan::NTuple{2, S} # Time span of calibration
    emissions::Vector{S} # Emissions data in GtCO₂ / year
    γ̂::Vector{S} # Observed growth rates γ̂
end

function timeinterpolation(t, tspan, v)
    tmin, tmax = tspan

    if t ≤ tmin return first(v) end
    if t ≥ tmax return last(v) end

    partition = range(tmin, tmax; length=length(v))
    udx = searchsortedfirst(partition, t)
    ldx = udx - 1

    α = (t - partition[ldx]) / step(partition)

    return (1 - α) * v[ldx] + α * v[udx]
end

"Growth rate of carbon concentration in the no-policy scenario `γₜ : [0, τ] -> [0, ∞)`."
function γ(_, calibration::ConstantCalibration)
    calibration.γ₀
end
function γ(t, calibration::DynamicCalibration)
    year = t + calibration.calibrationspan[1]
    return timeinterpolation(year, calibration.calibrationspan, calibration.γ̂)
end

"Linear interpolation of emissions in `calibration`"
function Eᵇ(t, calibration::DynamicCalibration)
    year = t + calibration.calibrationspan[1]
    return timeinterpolation(year, calibration.tspan, calibration.emissions)
end

Base.broadcastable(c::Calibration) = Ref(c)