using NeuralPDE
using Lux
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL

using ModelingToolkit
import ModelingToolkit: Interval, infimum, supremum
using IfElse

using Roots

using Plots
default(size = 600 .* (√2, 1), dpi = 180)


begin # Climate parameters from Mendez and Farazmand
    sectoyear = 3.154e7
    q₀ = 342
	η = 5.67e-8
	A, S = 20.5, 150
	cₚ = 280
	c₀ = 410
	κₓ = 5e8
	δ = 2.37e-10 * sectoyear # Decay per year

    α₁, α₂ = 0.31, 0.2 # Ice melting coefficients
	x₁, x₂ = 289, 295 # Temperatures at ice melting coefficients 
	xₐ = 3 # Transition rate

	h(x) = (1 + tanh(x / xₐ)) / 2
	Σ(x) = ((x - x₁) / (x₂ - x₁)) * h(x - x₁) * h(x₂ - x) + h(x - x₂)
	α(x) = α₁ * (1 - Σ(x)) + α₂ * Σ(x)
    μ(x, c) = q₀ * (1 - α(x)) + (S + A * log(c / cₚ)) - η * x^4
end

φ(c) = find_zeros(x -> μ(x, c), cₚ, 2cₚ)
x₀ = first(φ(c₀)) # Current Temperature

N = 101
cₗ, cₕ = c₀, 1.3c₀
carbonspace = range(cₗ, cₕ; length = 2N)

equilibriumtemperatures = φ.(carbonspace)
xₗ, xₕ = extrema(vcat(equilibriumtemperatures...))
temperaturespace = range(xₗ, xₕ; length = 2N)

ĉ = carbonspace[findfirst(x̄ -> length(x̄) < 2, φ.(carbonspace))]
x̂ = first(φ(ĉ))

begin # Economic parameters
    gigatonco2toppm = 7.821
    ρ = 0.01 # Discount rate
	τ = 0. # Carbont tax ($ ppm)
    β₀ = 97.1 / gigatonco2toppm # $ / ppm
    β₁ = 4.81 / gigatonco2toppm^2 # $ / (y ppm^2)
    γ = 10. # $ / K^2

	d(x) = γ * (x - x₀)^2 / 2 # Damages
	u(e) = (β₀ - τ) * e - (β₁ / 2) * e^2 # Profits

    emissionlimits = (1e-5, Inf)
    @assert emissionlimits[1] < δ * cₗ < emissionlimits[2] # make sure that steady state emissions are in range

    function e(pc) # emissions given shadow price of carbon stock
        min(max((β₀ - τ + pc) / β₁, emissionlimits[1]), emissionlimits[2])
    end

    l(x, e) = u(e) - d(x) # Payoff function
end

H(x, c, vx, vc) = l(x, e(vc)) + vx * κₓ * μ(x, c) + vc * (e(vc) - c * δ)

# Solution in single regime c > ĉ
@parameters x, c # Temperature, carbon
@variables v(..) # Value function in deviation from tipping point

Dx = Differential(x)
Dc = Differential(c)

hbj = H(x, c, Dx(v(x, c)), Dc(v(x, c))) ~ ρ * v(x, c)

# Neural network
dim = 2
k = 5
chain = Lux.Chain(Dense(dim, k, Lux.σ), Dense(k, k, Lux.σ), Dense(k, 1))

bc = [
    Dc(v(x̂, ĉ)) ~ β₁ * δ * ĉ - (β₀ - τ)
    v(x̂, ĉ) ~ l(x̂, δ * ĉ) / ρ
]


domains = [
    x ∈ Interval(x̂, maximum(temperaturespace)), 
    c ∈ Interval(ĉ, maximum(carbonspace))
]

dx = (maximum(temperaturespace) - x̂) / N
dc = (maximum(carbonspace) - ĉ) / N

discretization = PhysicsInformedNN(chain, GridTraining([dx, dc]))

@named pde = PDESystem(hbj, bc, domains, [x, c], [v(x, c)])
prob = discretize(pde, discretization)

callback = function (p, l)
    println("Loss = $l")
    return false
end

res = Optimization.solve(
    prob, Adam(0.001); 
    maxiters = 1000,
    callback = callback
) 

ϕ(x, c) = discretization.phi([x, c], res.u) |> first
contour(
    range(x̂, maximum(temperaturespace); length = 101), 
    range(ĉ, maximum(carbonspace); length = 101), 
    ϕ;
    levels = 20, fill = true, fillalpha = 0.5, color = :blues, xlabel = "Temperature", ylabel = "Carbon", title = "Value function")