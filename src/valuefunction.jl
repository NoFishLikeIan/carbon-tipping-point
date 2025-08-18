struct Time{T <: Real}
    τ::T
    t::Matrix{T}
end

struct PolicyState{T <: Real}
    policy::Matrix{Policy{T}}
    foc::Matrix{T}
end

struct ValueFunction{T <: Real}
    Fₜ::Matrix{T}
    Fₜ₊ₕ::Matrix{T}
    error::Matrix{T}
end

struct DPState{T <: Real}
    valuefunction::ValueFunction{T}
    policystate::PolicyState{T}
    timestate::Time{T}
end

DPState(calibration::Calibration{T}, G::RegularGrid{N, T}) where {N, T} = DPState(calibration.τ, G)
function DPState(τ::T, ::RegularGrid{N, T}) where {N, T}
    Fₜ = ones(T, (N, N))
    Fₜ₊ₕ = copy(Fₜ)
    error = similar(Fₜ)
    valuefunction = ValueFunction(Fₜ, Fₜ₊ₕ, error)

    policy = [ Policy{T}(0.5, 1.) for _ in 1:N, _ in 1:N ]
    foc = similar(error)
    policystate = PolicyState(policy, foc)    

    t = fill(τ, (N, N)) 

    return DPState(valuefunction, policystate, Time(τ, t))
end