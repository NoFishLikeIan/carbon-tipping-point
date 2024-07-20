struct LogUtility
    ρ::Float64  # Discount rate 
end

struct CRRA
    ρ::Float64  # Discount rate 
    θ::Float64  # Relative risk aversion
end

struct LogSeparable
    ρ::Float64  # Discount rate 
    θ::Float64  # Relative risk aversion
end

struct EpsteinZin
    ρ::Float64 # Discount rate 
    θ::Float64 # Relative Risk Aversion
    ψ::Float64 # Elasticity of Intertemporal Complementarity

    function EpsteinZin(ρ = 0.015, θ = 10., ψ = 1.)
        inelastic = ψ ≈ 1.
        timeadditive = ψ ≈ 1 / θ

        if inelastic && timeadditive
            LogUtility(ρ)
        elseif inelastic && !timeadditive
            LogSeparable(ρ, θ)
        elseif !inelastic && timeadditive
            CRRA(ρ, θ)
        else
            EpsteinZin(ρ, θ, ψ)
        end
    end
end

Preferences = Union{CRRA, EpsteinZin, LogUtility}

function f(c, v, Δt, p::EpsteinZin)
    ψ⁻¹ = 1 / p.ψ
    aggregator = (1 - p.θ) / (1 - ψ⁻¹)

    β = exp(-p.ρ * Δt)

    consumption = Δt * c^(1 - ψ⁻¹)
    value = β * ((1 - p.θ) * v)^inv(aggregator)

    return ((consumption + value)^aggregator) / (1 - p.θ)
end

"Climate damage aggregator. If θ > 1, it ought to be minimised!"
function g(χ, F, Δt, p::EpsteinZin)
    ψ⁻¹ = 1 / p.ψ
    aggregator = (1 - p.θ) / (1 - ψ⁻¹)

    β = exp(-p.ρ * Δt)

    consumption = (1 - β) * χ^(1 - ψ⁻¹) * Δt / (1 + p.ρ * Δt)
    value = β * F^inv(aggregator)

    (consumption + value)^aggregator
end

function f(c, v, Δt, p::CRRA)
    u = (c^(1 - p.θ)) / (1 - p.θ)

    βᵢ = exp(-p.ρ * Δt)

    βᵢ * v + Δt * u
end
function f(c, v, Δt, p::LogUtility)
    βᵢ = exp(-p.ρ * Δt)

    βᵢ * v + Δt * log(c)
end