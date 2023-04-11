@Base.kwdef struct OptimalPollution
    ρ::Float64 = 0.01 # Discount rate
    τ::Float64 = 0.0 # Emission tax
    γ::Float64 = 2.0 # Damage of temperature

    n::Int64 = 1 # Number of agents
    
    # Temperature process parameters
    c::Float64 = 0.01 # Temperature sensitivity
    σ²::Float64 = 1.0 # Volatility
    x̂::Float64 = 1.0 # Baseline temperature
end

"""
Optimal emissions given costate λ and model parameters.
"""
function E(λ, m::OptimalPollution)
    max(inv(m.τ - m.c * λ), 1e-5)
end

function V(x, x̂)
    (x - x̂)^4 / 4 - x̂^2 * (x - x̂)^2 / 2
end

"""
Deterministic evolution of temperature without emissions
"""
function μ(x, x̂)
    (x - x̂)^3 - x̂^2 * (x - x̂)
end

function μ′(x, x̂)
    3 * (x - x̂)^2 - 2 * x̂ * (x - x̂)
end

function d(x, x̂, γ)
    γ * exp(x - x̂)
end

function H(x, p, m::OptimalPollution)
    log(E(p, m)) - 1 - d(x, m.x̂, m.γ) - m.c * p * μ(x, m.x̂)
end

function Hₚ(x, p, m::OptimalPollution)
    m.c * (E(p, m) - μ(x, m.x̂))
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

function cumulemissions(x, ε, e::Function, m::OptimalPollution)
    if x ≤ m.x̂
        int, res = quadgk(y -> e(y, 1e-3), 0, x)

        if res > 1e-3
            @warn "Integration error too large: $res"
        end

        return int
    else
        int, res = quadgk(y -> e(y, 1e-3), m.x̂, x)

        if res > 1e-3
            @warn "Integration error too large: $res"
        end

        return cumulemissions(m.x̂, ε, e, m) + int
    end
end

function φ(x, ε, e::Function, m::OptimalPollution)
    C = exp(-m.c * V(x, m.x̂))

    return C * exp(m.c * cumulemissions(x, ε, e, m))
end