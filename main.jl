using DynamicalSystems
using LinearAlgebra
using ChaosTools

using Plots

include("src/socialplanner.jl")
include("utils/plotting.jl")
include("utils/dynamicalsystems.jl")

ρ = 0.01
η = 1 # logarithmic utility
c = 1 / 2
T₀ = 1.
γ = 0.5

p(τ, n) = [ρ, η, c, γ, T₀, n, τ]


ω(E, τ, η) = 1 - (E > 0 ? τ * E^η : 0)
ω′(E, τ, η) = E > 0 ? - τ * η / E^(1 - η) : 0

@inbounds function F!(dx, x, p, t)
    ρ, η, c, γ, T₀, n, τ = p
    T, E = x

    ΔT = T - T₀

    dx[1] = -c * (ΔT^3 - T₀^2 * ΔT - n * E)
    dx[2] = -E * ω(E, τ, η) * (ρ + c * (3 * ΔT^2 - T₀) + γ * T) / η
    
    return dx
end

@inbounds function DF!(J, x, p, t)
    ρ, η, c, γ, T₀, n, τ = p
    T, E = x

    ΔT = T - T₀

    J[1, 1] = -c * (3ΔT^2 - T₀^2)
    J[1, 2] = c * n
    J[2, 1] = -E * ω(E, τ, η) * (6c * ΔT + γ) / η
    J[2, 2] =  -(ω(E, τ, η) + E * ω′(E, τ, η)) * (ρ + c * (3 * ΔT^2 - T₀) + γ * T) / η

    return J
end

steadystates = [
    [0., 0.], 
    [2T₀, 0.]
]

emissions = range(-1, 2; length = 101)
temperature = range(-1, 2; length = 101)

ds(τ, n) = CoupledODEs(F!, zeros(2), p(τ, n))

basin_before_tax = computebasins(ds(0., 3), steadystates, (temperature, emissions))

basin_after_tax = computebasins(ds(0.8, 3), steadystates, (temperature, emissions))

P = tipping_probabilities(basin_before_tax, basin_after_tax)

heatmap(temperature, emissions, basin_before_tax)
heatmap(temperature, emissions, basin_after_tax)