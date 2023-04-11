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
function E(λ::Float64, m::OptimalPollution)
    max(inv(m.τ - m.c * λ), 0.)
end

"""
Deterministic evolution of temperature without emissions
"""
function μ(x::Float64, x̂::Float64)
    (x - x̂)^3 - x̂^2 * (x - x̂)
end

function μ′(x::Float64, x̂::Float64)
    3 * (x - x̂)^2 - 2 * x̂ * (x - x̂)
end

function d(x, x̂, γ)
    γ * exp(x - x̂)
end


"""
Deterministic skeleton of the optimal pollution model. 
"""
function deterministic(m::OptimalPollution)
    function h(state::NamedTuple, sol::NamedTuple)
        (; ρ, τ, c, x̂, γ)  = m
        (; v, vx_up) = sol
        x = state.x

        vx = vx_up

        emissions = max(τ - c * vx, 1e-5)

        vt = ρ * v + log(emissions) + c * vx * μ(x, x̂) + 1 + d(x, x̂, γ)

        return (vt = vt,)
    end
end

function firstordercorrection(m::OptimalPollution, v₀)
    v₀′(x) = v₀(x; ν = 1)
    v₀′′(x) = v₀(x; ν = 2)

    function h(state::NamedTuple, sol::NamedTuple)
        (; ρ, τ, c, x̂, γ, σ²)  = m
        (; v, vx_up) = sol
        x = state.x
        vx = vx_up

        emissions = max(τ - c * v₀′(x), 1e-5)

        vt = ρ * v - σ² * v₀′′(x) - vx * (inv(emissions) - c * μ(x, x̂))

        return (vt = vt,)
    end
end