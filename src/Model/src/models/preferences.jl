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

Preferences = Union{CRRA, LogSeparable, EpsteinZin}

function f(c, v, Δt, p::EpsteinZin)
    coeff = (1 - (1 / p.ψ)) / (1 - p.θ)
    value = ((1 - Θ) * v)^coeff

    δρ = exp(-p.ρ * Δt)

    u = δρ * value + (1 - δρ) * c^(1 - 1 / p.ψ)

    return u^inv(coeff) / (1 - p.θ)
end

function f(c, v, Δt, p::CRRA)
    δρ = exp(-p.ρ * Δt)

    u = (c^(1 - p.θ)) / (1 - p.θ) 

    return δρ * v + Δt * u
end

function f(χ, Xᵢ::Point, v, Δt, p::Preferences)
    c = χ * exp(Xᵢ.y)

    return f(c, v, Δt, p)
end