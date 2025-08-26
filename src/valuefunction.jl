mutable struct Time{S <: Real}
    t::S
end

"Finite Difference representation of the value function at time `t`"
struct ValueFunction{S <: Real, N₁, N₂}
    H::Matrix{S} # Matrix representation of the value function
    ε::Matrix{S} # Matrix represeting abatement
    t::Time{S}

    function ValueFunction(G::RegularGrid{N₁, N₂, S}, calibration::Calibration) where {N₁, N₂, S}
        t = Time(calibration.τ)
        H = ones(S, size(G))
        ε = ones(S, size(G))
        
        return new{S, N₁, N₂}(H, ε, t)
    end
end