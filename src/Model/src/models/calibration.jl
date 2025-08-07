struct Calibration{T <: Real}
    baselineyear::T # Baseline year of the calibration
    emissions::Vector{T} # Emissions data in GtCO₂ / year
    γparameters::NTuple{6, T} # Paramters for γ
    τ::T # End of calibration
end

struct RegionalCalibration{T <: Real}
    calibration::Calibration{T}
    fraction::Vector{T}
end

Base.broadcastable(c::Calibration) = Ref(c)
Base.broadcastable(c::RegionalCalibration) = Ref(c)

function γ(t, p::NTuple{6, T}) where T <: Real
    firstexp = p[1] * t * exp(-p[2] * t)
    secondexp = p[3] * exp(-p[4] * t)

    linear = p[5] + p[6] * t

    return firstexp + secondexp + linear
end

"Growth rate of carbon concentration in the no-policy scenario `γₜ : [0, τ] -> [0, ∞)`."
function γ(t, calibration::Calibration)
    γ(min(t, calibration.τ), calibration.γparameters)
end
function γ(t, regionalcalibration::RegionalCalibration)
    frac = interpolateovert(t, regionalcalibration.calibration.tspan, regionalcalibration.fraction)

    γₜ = γ(t, regionalcalibration.calibration)

    return γₜ * frac, γₜ * (1 - frac)
end


"Linear interpolation of emissions in `calibration`"
Eᵇ(t, calibration::Calibration) = interpolateovert(t, calibration.tspan, calibration.emissions)

function interpolateovert(t, tspan, v)

    tmin, tmax = tspan
    
    if t ≤ tmin return first(v) end
    if t ≥ tmax return last(v) end

    partition = range(tmin, tmax; length = length(v))
    udx = findfirst(tᵢ -> tᵢ > t, partition)
    ldx = udx - 1

    α = (t - partition[ldx]) / step(partition)

    return (1 - α) * v[ldx] + α * v[udx]
end