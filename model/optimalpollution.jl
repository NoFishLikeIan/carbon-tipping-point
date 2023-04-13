@Base.kwdef struct OptimalPollution
    ρ::Float64 = 0.01 # Discount rate
    τ::Float64 = 0.0 # Emission tax
    γ::Float64 = 2.0 # Damage of temperature
    
    # Temperature process parameters
    xᵤ::Float64 = 0.0 # unstable steady state 
    σ²::Float64 = 1.0 # Volatility
end

"""
Optimal emissions given costate λ and model parameters.
"""
function E(p, m::OptimalPollution)
    e = m.τ - p

    return e < 0 ? inv(1e-5) : inv(e)
end

function V(x, xᵤ)
    (x - xᵤ)^4 / 4 - xᵤ^2 * (x - xᵤ)^2 / 2
end

"""
Deterministic evolution of temperature without emissions
"""
function μ(x, xᵤ)
    (x - xᵤ)^3 - xᵤ^2 * (x - xᵤ)
end

function μ′(x, xᵤ)
    3 * (x - xᵤ)^2 - 2 * xᵤ * (x - xᵤ)
end

"""
Damage function. d(0, γ) = γ
"""
function d(x, xref, γ)
    exp(γ*(x - xref))
end

function H(x, p, m::OptimalPollution)
    log(E(p, m)) - 1 - d(x, -m.xᵤ, m.γ) - p * μ(x, m.xᵤ)
end

function Hₚ(x, p, m::OptimalPollution)
    E(p, m) - μ(x, m.xᵤ)
end

function Hₚₚ(x, p, m::OptimalPollution)
    m.c^2 * E(p, m)^2
end

"""
Deterministic skeleton of the optimal pollution model. 
"""
function deterministic(m::OptimalPollution)
    function h(state::NamedTuple, sol::NamedTuple)
        (; v, vx_up) = sol

        vt = m.ρ * v - H(state.x, vx_up, m)

        return (vt = vt,)
    end
end

function firstordercorrection(m::OptimalPollution, v₀)
    v₀′(x) = v₀(x; ν = 1)
    v₀′′(x) = v₀(x; ν = 2)

    function h(state::NamedTuple, sol::NamedTuple)
        (; v, vx_up) = sol
        x = state.x
        vx = vx_up

        vt = m.ρ * v - m.σ² * v₀′′(x) - Hₚ(x, v₀′(x), m) * vx

        return (vt = vt,)
    end
end

function secondordercorrection(m::OptimalPollution, v₀, v₁)
    v₀′(x) = v₀(x; ν = 1)
    v₁′(x) = v₁(x; ν = 1)
    v₁′′(x) = v₁(x; ν = 2)

    function h(state::NamedTuple, sol::NamedTuple)
        (; v, vx_up) = sol
        x = state.x
        vx = vx_up

        vt = m.ρ * v - m.σ² * v₁′′(x) - Hₚ(x, v₀′(x), m) * vx - Hₚₚ(x, v₀′(x), m) * v₁′(x) / 2

        return (vt = vt,)
    end
end