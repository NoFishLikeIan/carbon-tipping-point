struct Calibration
    years::Vector{Int} # Years of the ipcc data
    emissions::Vector{Float64} # Emissions in gton / year
    γparameters::NTuple{3, Float64} # Paramters for γ

    # Domain 
    tspan::NTuple{2, Float64} # Span of the IPCC data wrt to 2020
    r::Float64 # Decay rate for t > t₁
end

function γpol(t, calibration::Calibration)
    tmin = first(calibration.tspan)
    p = calibration.γparameters
    p[1] + p[2] * (t - tmin) + p[3] * (t - tmin)^2
end
"Parametric form of γ: (t₀, ∞) → [0, 1]"
function γ(t, calibration::Calibration)
    tmin, tmax = calibration.tspan
    p = calibration.γparameters
    Δt = min(t, tmax) - tmin

    pol = p[1] + p[2] * Δt + p[3] * Δt^2
    decay = exp(-calibration.r * max(0, t  - tmax))

    return pol * decay
end

"Linear interpolation of emissions in `calibration`"
Eᵇ(t, calibration::Calibration) = Eᵇ(t, calibration.tspan, calibration.emissions)
function Eᵇ(t, tspan, emissions)

    tmin, tmax = tspan
    
    if t ≤ tmin return first(emissions) end
    if t ≥ tmax return last(emissions) end

    partition = range(tmin, tmax; length = length(emissions))
    udx = findfirst(tᵢ -> tᵢ > t, partition)
    ldx = udx - 1

    α = (t - partition[ldx]) / step(partition)

    return (1 - α) * emissions[ldx] + α * emissions[udx]
end