function Base.show(io::IO, calibration::DynamicCalibration)
    println(io, "┌─ DynamicCalibration")

    t₀, t₁ = calibration.calibrationspan
    # Basic information
    println(io, "│  General Properties:")
    println(io, "│    ├─ Calibration span: [$(t₀), $(t₁)]")
    println(io, "│    ├─ Emission data points: $(length(calibration.emissions))")
    println(io, "│    └─ Growth rate points: $(length(calibration.γ̂))")

    # Generate Unicode plot of γ̂ (observed growth rates)
    println(io, "│")
    println(io, "│  Observed Growth Rates γ̂:")

    width = min(50, length(calibration.γ̂))
    idxs = round.(Int, range(1, length=width, stop=length(calibration.γ̂)))
    values = calibration.γ̂[idxs]

    min_val = minimum(values)
    max_val = maximum(values)

    chars = collect(" ▁▂▃▄▅▆▇█")
    range_val = max_val - min_val
    if range_val ≈ 0
        range_val = 1.0
    end

    indices = [round(Int, 1 + 8 * (val - min_val) / range_val) for val in values]
    indices = [max(1, min(9, idx)) for idx in indices]

    println(io, "│    max: $(round(max_val, digits=5))")
    print(io, "│    ")
    for idx in indices
        print(io, chars[idx])
    end
    println(io)
    println(io, "│    min: $(round(min_val, digits=5))")

    # Print x-axis labels
    print(io, "│    1")
    mid_point = round(Int, width/2)
    mid_spaces = max(0, mid_point - 1)
    print(io, " "^mid_spaces, "$(idxs[mid_point])")
    end_spaces = max(0, width - mid_point - 2)
    println(io, " "^end_spaces, "$(idxs[end])")

    println(io, "└─────────────────────────────────")
end
function Base.show(io::IO, ::MIME"text/plain", calibration::DynamicCalibration)
    show(io, calibration)
end