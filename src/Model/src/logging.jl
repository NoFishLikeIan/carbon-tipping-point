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
        return true
    else
        return isless(m₁.feedback.Tᶜ, m₂.feedback.Tᶜ)
    end
end