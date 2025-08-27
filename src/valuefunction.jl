mutable struct Time{S <: Real}
    t::S
end

"Finite Difference representation of the value function at time `t`"
struct ValueFunction{S <: Real, N₁, N₂}
    H::Matrix{S} # Matrix representation of the value function
    α::Matrix{S} # Matrix represeting abatement
    t::Time{S}

    function ValueFunction(hogg::Hogg, G::RegularGrid{N₁, N₂, S}, calibration::Calibration) where {N₁, N₂, S}
        t = Time(calibration.τ)
        H = ones(S, size(G))
        α = [ γ(calibration.τ, calibration) + δₘ(exp(Xᵢ.m) * hogg.Mᵖ, hogg) for Xᵢ in G.X ]
        
        return new{S, N₁, N₂}(H, α, t)
    end
end