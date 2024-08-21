struct Calibration
    years::Vector{Int} # Years of the ipcc data
    emissions::Vector{Float64} # Emissions in gton / year
    γparameters::NTuple{3, Float64} # Paramters for γ
    r::Float64 # Decay rate for t > t₁

    # Domain 
    tspan::NTuple{2, Float64} # Span of the IPCC data wrt to 2020
    r::Float64 # Decay rate for t > t₁
end

struct RegionalCalibration
    calibration::Calibration
    hiweights::Vector{Float64} # High income weighting
end

"Growth rate of carbon concentration in BAU"
function γ(t, calibration::Calibration)
    tmin, tmax = calibration.tspan
    p = calibration.γparameters
    Δt = min(t, tmax) - tmin

    pol = p[1] + p[2] * Δt + p[3] * Δt^2
    decay = exp(-calibration.r * max(0, t  - tmax))

    return pol * decay
end

"Growth rate of carbon concentration in BAU, regionally broke up"
function γ(t, rc::RegionalCalibration)
    γₜ = γ(t, rc.calibration)
    hw = weight(t, rc)

    return γₜ * hw, γₜ * (1 - hw)
end


"Linear interpolation of emissions in `calibration`"
Eᵇ(t, calibration::Calibration) = interpolateovert(t, calibration.tspan, calibration.emissions)
weight(t, rc::RegionalCalibration) = interpolateovert(t, rc.calibration.tspan, rc.hiweights)

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