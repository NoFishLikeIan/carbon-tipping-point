abstract type Calibration{S <: Real} end

Base.@kwdef struct ConstantCalibration{S} <: Calibration{S}
   γ₀::S = 0.022
end

"Growth rate of carbon concentration in the no-policy scenario `γₜ : [0, τ] -> [0, ∞)`."
function γ(_, calibration::ConstantCalibration)
    calibration.γ₀
end

struct DynamicCalibration{S} <: Calibration{S}
    calibrationspan::NTuple{2, S} # Time span of calibration
    emissions::Vector{S} # Emissions data in GtCO₂ / year
    γ̂::Vector{S} # Observed growth rates γ̂
end

function γ(t, calibration::DynamicCalibration)
    year = t + calibration.calibrationspan[1]
    return timeinterpolation(year, calibration.calibrationspan, calibration.γ̂)
end

struct DoubleExponentialCalibration{S} <: Calibration{S}
    calibrationspan::NTuple{2, S} # Time span of calibration
    emissions::Vector{S} # Emissions data in GtCO₂ / year
    γ₀::S
    α::S
    γ₁::S
    β::S
    γ̄::S
    γ̲::S
end

function γ(t, calibration::DoubleExponentialCalibration)
    @unpack γ₀, α, γ₁, β, γ̄, γ̲, calibrationspan = calibration
    t̄ = calibrationspan[2] - calibrationspan[1]
    τ = min(t, t̄)

    return max(γ₀ * exp(α * τ) + γ₁ * exp(β * τ) + γ̄, γ̲)
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

"Linear interpolation of emissions in `calibration`"
function Eᵇ(t, calibration::C) where {C <: Union{DynamicCalibration, DoubleExponentialCalibration}}
    year = t + calibration.calibrationspan[1]
    return timeinterpolation(year, calibration.calibrationspan, calibration.emissions)
end

Base.broadcastable(c::Calibration) = Ref(c)