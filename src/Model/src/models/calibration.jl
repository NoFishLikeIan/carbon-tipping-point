struct Calibration{T <: Real, N}
    baselineyear::T # Baseline year of the calibration
    emissions::Vector{T} # Emissions data in GtCO₂ / year
    γparameters::NTuple{N, T} # Paramters for γ
    τ::T # End of calibration
end

struct RegionalCalibration{T <: Real}
    calibration::Calibration{T}
    fraction::Vector{T}
end

Base.broadcastable(c::Calibration) = Ref(c)
Base.broadcastable(c::RegionalCalibration) = Ref(c)


"Growth rate of carbon concentration in the no-policy scenario `γₜ : [0, τ] -> [0, ∞)`."
function γ(t, p::NTuple{6})
    growth = p[1] * exp(p[2] * t)
    decay = p[3] * exp(p[4] * t)
    linear = p[5] + p[6] * t

    return linear + growth - decay
end
function γ(t, p::NTuple{8})
    num = p[1] * t^2 + p[2] * t + p[3]
    den = t^3 + p[4] * t^2 + p[5] * t + p[6]
    offset = p[7] * exp(-p[8] * t)
    
    return num / den + offset
end
function γ(t, calibration::Calibration)
    max(γ(min(t, 150), calibration.γparameters), 0)
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