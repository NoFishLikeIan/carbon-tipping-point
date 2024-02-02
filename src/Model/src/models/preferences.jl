Base.@kwdef struct LogUtility
    ρ::Float64 = 0.015  # Discount rate 
end


Base.@kwdef struct CRRA
    ρ::Float64 = 0.015  # Discount rate 
    θ::Float64 = 10.    # Relative risk aversion
end

Base.@kwdef struct LogSeparable
    ρ::Float64 = 0.015  # Discount rate 
    θ::Float64 = 10.    # Relative risk aversion
end

Base.@kwdef struct EpsteinZin
    ρ::Float64 = 0.015  # Discount rate 
    θ::Float64 = 10.    # Relative risk aversion
    ψ::Float64 = 1.5    # Elasticity of intertemporal complementarity 
end

Preferences = Union{CRRA, EpsteinZin, LogUtility}

function f(c, v, Δt, p::EpsteinZin)
    coeff = (1 - (1 / p.ψ)) / (1 - p.θ)
    value = ((1 - p.θ) * v)^coeff

    u = exp(-p.ρ * Δt) * value + Δt * c^(1 - 1 / p.ψ)

    return u^inv(coeff) / (1 - p.θ)
end

function f(c, v, Δt, p::CRRA)
    u = (c^(1 - p.θ)) / (1 - p.θ)

    βᵢ = exp(-p.ρ * Δt)

    return βᵢ * v + Δt * u
end

function f(c, v, Δt, p::LogUtility)
    βᵢ = exp(-p.ρ * Δt)

    return βᵢ * v + Δt * log(c)
end

function f(χ, Xᵢ::Point, v, Δt, p::Preferences)
    Y = exp(Xᵢ.y)
    c = χ * Y
    return f(c, v, Δt, p)
end