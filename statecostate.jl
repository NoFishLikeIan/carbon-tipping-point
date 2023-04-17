using DifferentialEquations
using Roots

begin
	using Plots
	default(size = 600 .* (√2, 1), dpi = 180)
end

begin # Climate parameters from Mendez and Farazmand
    sectoyear = 3.154e7
    q₀ = 342
	η = 5.67e-8
	A, S = 20.5, 150
	cₚ = 280
	c₀ = 410
	κₓ = 1
	δ = 2.37e-10 * sectoyear # Decay per year

    α₁, α₂ = 0.31, 0.2 # Ice melting coefficients
	x₁, x₂ = 289, 295 # Temperatures at ice melting coefficients 
	xₐ = 3 # Transition rate

	h(x) = (1 + tanh(x / xₐ)) / 2
	h′(x) = (1 / 2xₐ) * sech(x / xₐ)^2

	Σ(x) = ((x - x₁) / (x₂ - x₁)) * h(x - x₁) * h(x₂ - x) + h(x - x₂)
	Σ′(x) = h′(x - x₂) + (
		h(x - x₁) * h(x₂ - x) +
		(x - x₁) * h′(x - x₁) * h(x₂ - x) -
		(x - x₁) * h(x - x₁) * h′(x₂ - x)
	) / (x₂ - x₁)

	α(x) = α₁ * (1 - Σ(x)) + α₂ * Σ(x)
	α′(x) = - (α₁ - α₂) * Σ′(x)

    μ(x, c) = q₀ * (1 - α(x)) + (S + A * log(c / cₚ)) - η * x^4
	
	∂xμ(x, c) = - q₀ * α′(x) - 4 * η * x^3
	∂cμ(x, c) = A / c
end

begin # Economic parameters
    gigatonco2toppm = 7.821
    ρ = 0.01 # Discount rate
	τ = 0. # Carbont tax ($ ppm)
    β₀ = 97.1 / gigatonco2toppm # $ / ppm
    β₁ = 4.81 / gigatonco2toppm^2 # $ / (y ppm^2)
    γ = 1. # $ / K^2
	
	d(x) = γ * x^2 / 2 # Damages
	u(e) = (β₀ - τ) * e - (β₁ / 2) * e^2 # Profits
	
    emissionlimits = (1e-5, Inf)
    @assert emissionlimits[1] < δ * cₗ < emissionlimits[2] # make sure that steady state emissions are in range
	
	# emissions given shadow price of carbon stock
    e(pc) = max((β₀ - τ + pc) / β₁, 0.)
    l(x, e) = u(e) - d(x) # Payoff function
end

# Make regions
allconcentrations = range(100, 500; length = 1001)
φ(c) = find_zeros(x -> μ(x, c), 0.01, 1000.) # Equilibrium temperatures associated with c

equiltemperatures = φ.(allconcentrations)

idx, jdx = findfirst(x -> length(x) > 1, equiltemperatures), findlast(x -> length(x) > 1, equiltemperatures)

cₗ = allconcentrations[idx]
cᵤ = allconcentrations[jdx]

xₗ = minimum(equiltemperatures[idx])
xᵤ = maximum(equiltemperatures[jdx])

x₀ = first(φ(c₀))

function f(du, u, p, t)
	x, c, λx, λc = u
	
	du[1] = κₓ * μ(x, c)
	du[2] = e(λc) - δ * c

	du[3] = (ρ + δ) * λc - ∂cμ(x, c) * λx * κₓ
	du[4] = (ρ + ∂xμ(x, c) * κₓ) * λx - γ * x

end

timespan = (0, 100) # years
dt = 1 / 365

prob = ODEProblem(f, [x₀, c₀, -10., -10.], timespan, dt = dt)
sol = solve(prob, Tsit5())

time = range(timespan[1], timespan[2]; step = dt)
plot(time, t -> sol(t)[1]; xlabel = "Time (years)", ylabel = "Temperature (K)", c = :darkblue, yguidefontcolor = :darkblue,  label = false, rightmargin = 10Plots.mm)

plot!(twinx(), time, t -> sol(t)[2]; c = :darkred, yguidefontcolor = :darkred, ylabel =  "Concentration", label = false)

