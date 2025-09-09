mutable struct Time{S <: Real}
    t::S
end

"Finite Difference representation of the value function at time `t`"
struct ValueFunction{S <: Real, N₁, N₂}
    H::Matrix{S} # Matrix representation of the value function
    α::Matrix{S} # Matrix represeting abatement
    t::Time{S}

    function ValueFunction(climate::C, G::GR, calibration::Calibration) where {N₁, N₂, S, GR <: AbstractGrid{N₁, N₂, S}, C <: Climate{S}}
        ValueFunction(calibration.τ, climate, G, calibration)
    end
    function ValueFunction(τ, climate::C, G::GR, calibration::Calibration) where {N₁, N₂, S, GR <: AbstractGrid{N₁, N₂, S}, C <: Climate{S}}
        t = Time(τ)
        H = ones(S, size(G))
        α = [ γ(τ, calibration) + δₘ(exp(x.m) * climate.hogg.Mᵖ, climate.decay) for x in G ]
        
        return new{S, N₁, N₂}(H, α, t)
    end

    function ValueFunction{S, N₁, N₂}(H, α, t) where {S, N₁, N₂}
        new{S, N₁, N₂}(H, α, t)
    end
end

# Base.show method for ValueFunction
function Base.show(io::IO, vf::ValueFunction{S, N₁, N₂}) where {S, N₁, N₂}
    println(io, "┌─ ValueFunction{$(N₁)×$(N₂), $(S)}")
    
    # Print time information
    println(io, "│  Time: $(round(vf.t.t, digits=3))")
    
    # Print value function statistics
    H_min, H_max = extrema(vf.H)
    H_mean = sum(vf.H) / length(vf.H)
    println(io, "│  Value Function (H):")
    println(io, "│    ├─ Range: [$(round(H_min, digits=4)), $(round(H_max, digits=4))]")
    println(io, "│    └─ Mean: $(round(H_mean, digits=4))")
    
    # Print abatement statistics
    α_min, α_max = extrema(vf.α)
    α_mean = sum(vf.α) / length(vf.α)
    println(io, "│  Abatement (α):")
    println(io, "│    ├─ Range: [$(round(α_min, digits=4)), $(round(α_max, digits=4))]")
    println(io, "│    └─ Mean: $(round(α_mean, digits=4))")
    
    # Print matrix dimensions
    println(io, "│  Matrix size: $(size(vf.H))")
    
    println(io, "└─────────────────────────────────")
end
function Base.show(io::IO, ::MIME"text/plain", vf::ValueFunction)
    show(io, vf)
end