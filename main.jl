using DifferentialEquations

include("src/socialplanner.jl")
include("utils/plotting.jl")

m = SocialPlanner()
p = [m.η, m.ρ, m.τ, m.d, m.b, m.a, m.σ², m.Tᵣ] # Parameter space

function g(λ, T)
    dx = zeros(2)
    statecostate!(dx, [λ, T], p, 0.0)

    return dx
end

Λ = (-2, 2)
T = (-2, 2)

narrows = 25
plotvectorfield(
    range(Λ...; length=narrows), range(T...; length=narrows),
    g; aspect_ratio=(Λ[2] - Λ[1]) / (T[2] - T[1]), cmap=:viridis, alpha=0.5
)