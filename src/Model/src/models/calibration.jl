struct Calibration{N}
    baselineyear::Float64 # Baseline year of the calibration
    emissions::Vector{Float64} # Emissions data in GtCO₂ / year
    γparameters::NTuple{N, Float64} # Paramters for γ
    τ::Float64 # End of calibration
end

struct RegionalCalibration
    calibration::Calibration
    fraction::Vector{Float64}
end

Base.broadcastable(c::Calibration) = Ref(c)
Base.broadcastable(c::RegionalCalibration) = Ref(c)

"Growth rate of carbon concentration in BAU"
function γ(t, calibration::Calibration{2})
    year = calibration.baselineyear + min(t, calibration.τ)
    p = calibration.γparameters

    return p[1] + p[2] * year
end
function γ(t, calibration::Calibration{3})
    year = calibration.baselineyear + min(t, calibration.τ)
    p = calibration.γparameters

    return p[1] + p[2] * year + p[3] * year^2
end
function γ(t, calibration::Calibration{4})
    year = calibration.baselineyear + min(t, calibration.τ)
    p = calibration.γparameters

    return p[1] + p[2] * year + p[3] * year^2 + p[4] * year^3
end
function γ(t, calibration::Calibration{5})
    year = calibration.baselineyear + min(t, calibration.τ)
    p = calibration.γparameters

    return p[1] + p[2] * year + p[3] * year^2 + p[4] * year^3 + p[5] * year^4
end
function γ(t, calibration::Calibration{6})
    year = calibration.baselineyear + min(t, calibration.τ)
    p = calibration.γparameters

    return p[1] + p[2] * year + p[3] * year^2 + p[4] * year^3 + p[5] * year^4 + p[6] * year^5
end
function γ(t, calibration::Calibration{7})
    year = calibration.baselineyear + min(t, calibration.τ)
    p = calibration.γparameters

    return p[1] + p[2] * year + p[3] * year^2 + p[4] * year^3 + p[5] * year^4 + p[6] * year^5 + p[7] * year^6
end
function γ(t, calibration::Calibration{8})
    year = calibration.baselineyear + min(t, calibration.τ)
    p = calibration.γparameters

    return p[1] + p[2] * year + p[3] * year^2 + p[4] * year^3 + p[5] * year^4 + p[6] * year^5 + p[7] * year^6 + p[8] * year^7
end
function γ(t, calibration::Calibration{9})
    year = calibration.baselineyear + min(t, calibration.τ)
    p = calibration.γparameters

    return p[1] + p[2] * year + p[3] * year^2 + p[4] * year^3 + p[5] * year^4 + p[6] * year^5 + p[7] * year^6 + p[8] * year^7 + p[9] * year^8
end
function γ(t, calibration::Calibration{10})
    year = calibration.baselineyear + min(t, calibration.τ)
    p = calibration.γparameters

    return p[1] + p[2] * year + p[3] * year^2 + p[4] * year^3 + p[5] * year^4 + p[6] * year^5 + p[7] * year^6 + p[8] * year^7 + p[9] * year^8 + p[10] * year^9
end
function γ(t, calibration::Calibration{11})
    year = calibration.baselineyear + min(t, calibration.τ)
    p = calibration.γparameters

    return p[1] + p[2] * year + p[3] * year^2 + p[4] * year^3 + p[5] * year^4 + p[6] * year^5 + p[7] * year^6 + p[8] * year^7 + p[9] * year^8 + p[10] * year^9 + p[11] * year^10
end
function γ(t, calibration::Calibration{12})
    year = calibration.baselineyear + min(t, calibration.τ)
    p = calibration.γparameters

    return p[1] + p[2] * year + p[3] * year^2 + p[4] * year^3 + p[5] * year^4 + p[6] * year^5 + p[7] * year^6 + p[8] * year^7 + p[9] * year^8 + p[10] * year^9 + p[11] * year^10 + p[12] * year^11
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