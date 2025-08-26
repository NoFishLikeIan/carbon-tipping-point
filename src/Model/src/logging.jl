function Base.show(io::IO, model::AbstractModel)
    # Get the type name without parameters
    model_type = split(string(typeof(model)), "{")[1]
    
    println(io, "┌─ $(model_type)")
    
    # Print climate parameters
    try
        if model isa TippingModel
            ecs_vals = ecs(model.hogg, model.feedback)
            if length(ecs_vals) == 1
                println(io, "│  Climate Parameters:")
                println(io, "│    ├─ ECS: $(round(only(ecs_vals), digits=2)) °C")
            else
                println(io, "│  Climate Parameters:")
                println(io, "│    ├─ ECS: $(join(["$(round(v, digits=2)) °C" for v in ecs_vals], ", "))")
            end
        else
            ecs_val = ecs(model.hogg)
            println(io, "│  Climate Parameters:")
            println(io, "│    ├─ ECS: $(round(ecs_val, digits=2)) °C")
        end
        println(io, "│    ├─ Temp volatility: $(round(model.hogg.σₜ, digits=4))")
        println(io, "│    └─ CO₂ volatility: $(round(model.hogg.σₘ, digits=4))")
    catch
        println(io, "│  Climate Parameters: [Error computing ECS]")
        println(io, "│    ├─ Temp volatility: $(round(model.hogg.σₜ, digits=4))")
        println(io, "│    └─ CO₂ volatility: $(round(model.hogg.σₘ, digits=4))")
    end
    
    # Print damage function
    damage_type = split(string(typeof(model.damages)), "{")[1]
    println(io, "│  Damage Function: $(damage_type)")
    
    if model.damages isa WeitzmanLevel
        println(io, "│    └─ ξ = $(round(model.damages.ξ, digits=6))")
    elseif model.damages isa Kalkuhl
        println(io, "│    ├─ ξ₁ = $(round(model.damages.ξ₁, digits=4))")
        println(io, "│    └─ ξ₂ = $(round(model.damages.ξ₂, digits=4))")
    elseif model.damages isa WeitzmanGrowth
        println(io, "│    ├─ ξ = $(round(model.damages.ξ, digits=6))")
        println(io, "│    └─ ν = $(round(model.damages.ν, digits=2))")
    end
    
    # Print preferences
    pref_type = split(string(typeof(model.preferences)), "{")[1]
    println(io, "│  Preferences: $(pref_type)")
    
    if model.preferences isa EpsteinZin
        println(io, "│    ├─ ρ = $(round(model.preferences.ρ * 100, digits=2))%")
        println(io, "│    ├─ θ = $(round(model.preferences.θ, digits=2))")
        println(io, "│    └─ ψ = $(round(model.preferences.ψ, digits=2))")
    elseif model.preferences isa LogSeparable || model.preferences isa CRRA
        println(io, "│    ├─ ρ = $(round(model.preferences.ρ * 100, digits=2))%")
        println(io, "│    └─ θ = $(round(model.preferences.θ, digits=2))")
    elseif model.preferences isa LogUtility
        println(io, "│    └─ ρ = $(round(model.preferences.ρ * 100, digits=2))%")
    end
    
    # Print model-specific parameters
    if model isa TippingModel
        println(io, "│  Tipping Point:")
        println(io, "│    ├─ Critical temp: Tᶜ = $(round(model.feedback.Tᶜ - model.hogg.Tᵖ, digits=2)) °C")
        println(io, "│    ├─ Strength: ΔS = $(round(model.feedback.ΔS, digits=2))")
        println(io, "│    └─ Steepness: L = $(round(model.feedback.L, digits=2))")
    elseif model isa JumpModel
        println(io, "│  Jump Process:")
        println(io, "│    ├─ Base intensity: i₀ = $(round(model.jump.i₀, digits=2))")
        println(io, "│    └─ Temp sensitivity: i₁ = $(round(model.jump.i₁, digits=2))")
    end
    
    println(io, "└─────────────────────────────────")
end

# Add support for display in notebooks and REPLs
function Base.show(io::IO, ::MIME"text/plain", model::AbstractModel)
    show(io, model)
end

function Base.isless(m₁::M₁, m₂::M₂) where {M₁ <: AbstractModel, M₂ <: AbstractModel}
    if (m₁ isa FirstOrderLinearModel) || (m₂ isa FirstOrderLinearModel)
        return false
    else
        return isless(m₁.feedback.Tᶜ, m₂.feedback.Tᶜ)
    end
end


function Base.show(io::IO, calibration::Calibration{T,N}) where {T,N}
    println(io, "┌─ Calibration")
    
    # Basic information
    println(io, "│  General Properties:")
    println(io, "│    ├─ Base year: $(calibration.baselineyear)")
    println(io, "│    ├─ Time horizon: $(calibration.τ) years")
    println(io, "│    └─ Emission data points: $(length(calibration.emissions))")
    
    # γ parameters
    println(io, "│  Growth Rate Parameters:")
    for (i, p) in enumerate(calibration.γparameters)
        if i == N
            println(io, "│    └─ p$i = $(round(p, digits=6))")
        else
            println(io, "│    ├─ p$i = $(round(p, digits=6))")
        end
    end
    
    # Generate Unicode plot of γ function
    println(io, "│")
    println(io, "│  Growth Rate Function γ(t):")
    
    # Sample the function
    width = 50  # Width of the plot
    ts = range(0, calibration.τ, length=width)
    values = [γ(t, calibration) for t in ts]
    
    # Calculate min/max for scaling
    min_val = minimum(values)
    max_val = maximum(values)
    
    # Create Unicode sparkline
    chars = collect(" ▁▂▃▄▅▆▇█")
    range_val = max_val - min_val
    if range_val ≈ 0
        range_val = 1.0
    end
    
    # Map values to character indices
    indices = [round(Int, 1 + 8 * (val - min_val) / range_val) for val in values]
    indices = [max(1, min(9, idx)) for idx in indices]
    
    # Print the sparkline with labels
    println(io, "│    max: $(round(max_val, digits=5))")
    print(io, "│    ")
    for idx in indices
        print(io, chars[idx])
    end
    println(io)
    println(io, "│    min: $(round(min_val, digits=5))")
    
    # Print x-axis labels
    print(io, "│    0")
    mid_point = round(Int, width/2)
    mid_spaces = max(0, mid_point - 1)
    print(io, " "^mid_spaces, "$(round(Int, ts[mid_point]))")
    
    end_spaces = max(0, width - mid_point - 2)
    println(io, " "^end_spaces, "$(round(Int, ts[end]))")
    
    println(io, "└─────────────────────────────────")
end

# Add support for display in notebooks and REPLs
function Base.show(io::IO, ::MIME"text/plain", calibration::Calibration)
    show(io, calibration)
end

# Handle RegionalCalibration with two regions
function Base.show(io::IO, calibration::RegionalCalibration{T}) where T
    println(io, "┌─ RegionalCalibration")
    
    # Show underlying calibration
    println(io, "│  Underlying Calibration:")
    println(io, "│    ├─ Base year: $(calibration.calibration.baselineyear)")
    println(io, "│    ├─ Time horizon: $(calibration.calibration.τ) years")
    println(io, "│    └─ Emission data points: $(length(calibration.calibration.emissions))")
    
    # Regional fractions
    println(io, "│  Regional Data:")
    println(io, "│    ├─ Regions: 2")
    println(io, "│    └─ Current fraction: $(round(calibration.fraction[end], digits=3))")
    
    # Generate Unicode plot for both regions
    println(io, "│")
    println(io, "│  Regional Growth Rate Functions:")
    
    # Sample the functions
    width = 50
    ts = range(0, calibration.calibration.τ, length=width)
    
    # Get values for both regions
    values_by_region = [γ(t, calibration) for t in ts]
    values_region1 = [val[1] for val in values_by_region]
    values_region2 = [val[2] for val in values_by_region]
    
    # Find overall min/max
    all_values = vcat(values_region1, values_region2)
    min_val = minimum(all_values)
    max_val = maximum(all_values)
    
    # Create sparklines
    chars = " ▁▂▃▄▅▆▇█"
    range_val = max_val - min_val
    if range_val ≈ 0
        range_val = 1.0
    end
    
    # Map values to character indices for both regions
    indices1 = [round(Int, 1 + 8 * (val - min_val) / range_val) for val in values_region1]
    indices1 = [max(1, min(9, idx)) for idx in indices1]
    
    indices2 = [round(Int, 1 + 8 * (val - min_val) / range_val) for val in values_region2]
    indices2 = [max(1, min(9, idx)) for idx in indices2]
    
    # Print sparklines
    println(io, "│    max: $(round(max_val, digits=5))")
    
    print(io, "│    R1: ")
    for idx in indices1
        print(io, chars[idx])
    end
    println(io)
    
    print(io, "│    R2: ")
    for idx in indices2
        print(io, chars[idx])
    end
    println(io)
    
    println(io, "│    min: $(round(min_val, digits=5))")
    
    # Print x-axis labels
    print(io, "│    0")
    mid_point = round(Int, width/2)
    mid_spaces = max(0, mid_point - 1)
    print(io, " "^mid_spaces, "$(round(Int, ts[mid_point]))")
    
    end_spaces = max(0, width - mid_point - 2)
    println(io, " "^end_spaces, "$(round(Int, ts[end]))")
    
    println(io, "└─────────────────────────────────")
end

function Base.show(io::IO, ::MIME"text/plain", calibration::RegionalCalibration)
    show(io, calibration)
end

