struct Calibration
    years::Vector{Int} # Years of the ipcc data
    emissions::Vector{Float64} # Emissions in gton / year
    γparameters::NTuple{3, Float64} # Paramters for γ

    # Domain 
    tspan::NTuple{2, Float64} # Span of the IPCC data wrt to 2020
end


"Parametric form of γ: (t₀, ∞) → [0, 1]"
γ(t, calibration::Calibration) = γ(t, calibration.γparameters, first(calibration.tspan))
function γ(t, p, t₀)
   max(p[1] + p[2] * (t - t₀) + p[3] * (t - t₀)^2, 0.)
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