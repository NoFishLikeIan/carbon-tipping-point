### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 83e33f5e-354d-4296-aafc-e8778d4839fb
using DrWatson

# ╔═╡ 14b581f5-a235-47d2-baf3-065cffff6f51
@quickactivate "scc-tipping-points"

# ╔═╡ e80c5fa0-dad7-11ed-22a9-c7d0f0e8f71f
using DifferentialEquations, DynamicalSystems

# ╔═╡ 585cee5d-0ac8-49fc-b760-9200972e30a1
using Roots

# ╔═╡ 67b6ee24-192b-4b19-b876-757087f26263
using PlutoUI

# ╔═╡ 09589e67-c55e-44a5-bd51-37a53e8a8585
begin
	using Plots
	default(size = 600 .* (√2, 1), dpi = 300, margins = 5Plots.mm, linewidth = 1.5)
end

# ╔═╡ c1fc4206-5541-4c8e-92f3-7e072fd17a5d
using LinearAlgebra

# ╔═╡ 9bc05e95-a185-43b6-afe6-4707d4cefc2f
using ForwardDiff

# ╔═╡ 24b43a3d-d657-44e4-8aca-995ee007ebb2
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
	}
</style>
"""


# ╔═╡ 94078a8e-c9ee-4351-8d4d-c9656024934d
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ fe8e8cc9-8547-4eba-8120-e4d6632fe9b8


# ╔═╡ c5287aba-1465-46ea-bdf8-7d9cf19bf945
begin
	function plotvectorfield(xs, ys, g::Function; plotkwargs...)
	    fig = plot()
	    plotvectorfield!(fig, xs, ys, g; plotkwargs...)
	    return fig
	end
	
	function plotvectorfield!(figure, xs, ys, g::Function; rescale = 1, plotkwargs...)
	
		xlims = extrema(xs)
		ylims = extrema(ys)
		
		N, M = length(xs), length(ys)
		xm = repeat(xs, outer=M)
		ym = repeat(ys, inner=N)
		
		field = g.(xm, ym)
		
		scale = rescale * (xlims[2] - xlims[1]) / min(N, M)
		u = @. scale * first(field)
		v = @. scale * last(field)
		
		steadystates = @. (u ≈ 0) * (v ≈ 0)
		
		u[steadystates] .= NaN
		v[steadystates] .= NaN
		
		z = (x -> √(x'x)).(field)
		
	    quiver!(
	        figure, xm, ym;
	        quiver = (u, v), line_z=repeat(z, inner=4),
	        aspect_ratio = 1, xlims = xlims, ylims = ylims,
	        c = :batlow, colorbar = false,
	        plotkwargs...
	    )
	
	end
end

# ╔═╡ 61b868ce-b080-471d-8de8-edf807b253ce
climate = ingredients("../model/climate.jl");

# ╔═╡ 8d16ec1e-e16f-45d4-93a4-1cc1a18fb487
economy = ingredients("../model/economic.jl");

# ╔═╡ 241e7857-3178-44de-9b4e-85f0b717d131
dsutils = ingredients("../utils/dynamicalsystems.jl");

# ╔═╡ b78704bf-bcc4-4859-8978-fcde1a64dafb
gigatonco2toppm = 7.821;

# ╔═╡ 519620d9-c504-40ff-998e-262dbe3caeb0
sectoyear = 3.154e7;

# ╔═╡ 3fe60d4b-6d9d-4dbb-9f99-cbae81ef6cd3
md"## Climate model from Mendez and Farazmand (2021)"

# ╔═╡ ff62d855-fccf-4418-b255-333832c58f29
md"
Originally the paper uses the function $\Sigma$ function to model the ice melting coefficients. I will approximate this by $\sigma$
"

# ╔═╡ 6fcf3aca-84a6-4d71-a498-fff91ffae7f2
# ╠═╡ disabled = true
#=╠═╡
m = climate.MendezFarazmand();
  ╠═╡ =#

# ╔═╡ bd97a765-f358-4400-b013-3c592bd864fe
begin
	a(x, m) = m.q₀ * ((1 - m.α₁) + (m.α₁ - m.α₂) * climate.σ(x, m))
	a′(x, m) = m.q₀ * (m.α₁ - m.α₂) * climate.σ′(x, m)
	a′′(x, m) = m.q₀ * (m.α₁ - m.α₂) * climate.σ′′(x, m)
end;

# ╔═╡ 5a2c219a-a727-44dd-917b-759667911d77
let
	(; α₁, α₂, x₁, x₂, q₀) = m
	
	h(x) = (1 + tanh(x / 3)) / 2
	Σ(x) = ((x - m.x₁) / (m.x₂ - m.x₁)) * h(x - m.x₁) * h(m.x₂ - x) + h(x - m.x₂)
	
	xs = range(280, 310; length = 101)
	
	basefig = plot(xlabel = "Temperature (Kelvin) \$x\$", ylabel = "Baseline temperature", dpi = 300, legend = :bottomright)
	
	plot!(basefig, xs, x -> q₀ * ((1 - α₁) + (α₁ - α₂) * Σ(x)); c = :darkblue, label = "Mendez and Farazmand")	
	
	vline!(basefig, [m.x₁]; label = false, c = :black, linestyle = :dash)
	vline!(basefig, [m.x₂]; label = false, c = :black, linestyle = :dash)
	vline!(basefig, [(m.x₂ + m.x₁) / 2]; label = false, c = :black, linestyle = :solid)

	plot!(basefig, xs, x -> a(x, m); label = "This paper", c = :darkred)

	savefig(basefig, "../docs/figures/baseline-temperature.png")

	basefig
end

# ╔═╡ 82d4835a-2a2f-4208-ad23-9e4c9ac2cba0
current_emissions = 2.59 + m.δ * m.c₀

# ╔═╡ 54b0c489-3c22-4206-8af3-d4cc8a372c17
tipping_points = find_zeros(x -> a′(x, m) - 4m.η * x^3, (290, 300))

# ╔═╡ ad1738b0-8f7c-4c91-92a6-ad4bc096a98d
md"## Economic model"

# ╔═╡ 60a24b1b-d933-45bb-8638-9167ab9dd520
Llinear = economy.LinearQuadratic(γ = 0.5, τ = 0., xₛ = 280);

# ╔═╡ eca60d2f-b7c1-4153-8a1f-268b94f66a2a
md"
## Linear System

A simplified linear model

- Climate damages $\gamma_c$ $(@bind γc Slider(0:0.001:1, show_value = true, default = 0.01))

"

# ╔═╡ 85d4cad2-9f17-41cf-9b27-d3accb7057e8
begin
	(; κ, A, δ) = m
	(; β₀, β₁, ρ, τ, xₛ) = Llinear
	
	cₛ = m.cₚ
	eᵤ = (β₀ - τ) / β₁
	
	A = [
		ρ + δ 	γc / β₁;
		1 		-δ;
	]

	ηₛ = 1/2 * (ρ - √(ρ^2 + 4 * (ρ + δ) * δ + γc / β₁))

	ubase = inv(A) * [(ρ + δ) * eᵤ; (γc / β₁) * cₛ]

	λ, V = eigen(A)

	optimalemissions(c) = inv(δ + ηₛ) * c

end

# ╔═╡ 314c6433-27ca-4aba-bd41-0e80f5c1118f
let
	(; δ, cₚ) = m
	(; ρ, β₁, β₀) = Llinear
	c̄(γ, τ) = (β₀ - τ - γ * cₚ) / (β₁ * δ + γ / (ρ + δ))

	γspace = range(0.0001, 0.0008; length = 1001)
	τspace = range(0., 12.; length = 1001) * (gigatonco2toppm / sectoyear) * 1e9
	
	css = contour(γspace, τspace, (γ, τ) -> c̄(γ, sectoyear * τ / (gigatonco2toppm * 1e9)); c = :viridis, contourlabels = true, levels = cₚ:150:1500, cbar = false, ylabel = "Carbon tax \$ \\tau \$ in USD per ton of carbon", xlabel = "Damage coefficient \$ \\gamma_c \$", dpi = 300)

	# savefig(css, "../docs/figures/linear-cbar.png")
	
	css
end

# ╔═╡ b17c0b74-40a4-4d90-a664-7025c44fef06
md"
## State costate
"

# ╔═╡ f3bd2d72-5b85-48b9-982e-782d096d5e4e
γ₀ = 0.000751443;

# ╔═╡ 4f9964b2-05fa-4c01-b003-7a3ca32041b4
begin
	c₀ = 410 # Current carbon concentration
	x₀ = 289 # Current temperature
end;

# ╔═╡ 7b56773e-7d5f-411e-a5de-541ba6e7e0b0
let
	xspace = range(285, 297; length = 101)
	css = (x -> climate.φ(x, m)).(xspace)

	arrows = 17
	yarrows = range(extrema(xspace)...; length = arrows)
	xarrows = range(extrema(css)...; length = arrows)
	
	ssfig = plotvectorfield(
		xarrows, yarrows, (c, x) -> [climate.μ(x, c, m); current_emissions - m.δ * c ];
		xlims = extrema(css), ylims = extrema(xspace), 
		xlabel = "Carbon concentration \$c\$", ylabel = "Temperature \$x\$",
		aspect_ratio = (maximum(css) - minimum(css)) / (xspace[end] - xspace[1]),
		rescale = 0.0001, c = :coolwarm, dpi = 300
	)


	plot!(ssfig, css, xspace; label = "\$\\mu(x, c) = 0\$", c = :darkred)
	scatter!(ssfig, [m.c₀], [x₀]; c = :black, label = "Current \$(c_0, x_0)\$", marker = 3.5)

	hline!(ssfig, tipping_points; label = "Tipping points", c = :black, linestyle = :dash)

	# savefig(ssfig, "../docs/figures/temperature-dynamics.png")

	ssfig
		
end

# ╔═╡ cae95c3f-698c-4287-8c68-c91186c63dc0
let
	(; β₀, τ, β₁, xₛ, γ) = Llinear
	
	b(e, x) = (β₀ - τ) * e - (β₁ / 2) * e^2 - (γ / 2) * (x - xₛ)^2 

	xspace = range(285, 297; length = 101)
	espace = range(-20, 20; length = 101)

	lfig = contourf(
		espace, xspace, b; 
		ylabel = "Temperature \$x\$", xlabel = "Emissions \$e\$",
		c = :viridis, 
		levels = 10, linewidth = 0,
		aspect_ratio =  (espace[end] - espace[1]) / (xspace[end] - xspace[1]),
		xlims = extrema(espace), ylims = extrema(xspace), dpi = 300
	)

	scatter!(lfig, [current_emissions], [x₀]; c = "black", label = "Current emissions and temperature")

	# savefig(lfig, "../docs/figures/benefit-functional.png")
	
	lfig
end

# ╔═╡ 9f888b0c-7d8b-4213-8820-f584b3a36500
let
	
	narrows = 22
	cspace = range(-300, 300; length = narrows)
	emissionspace = range(-300, 300; length = narrows)

	aspect_ratio = (emissionspace[end] - emissionspace[1]) / (cspace[end] - cspace[1]) 

	
	vecfig = plotvectorfield(
		emissionspace, cspace, (e, c) -> A * [e; c]; 
		aspect_ratio = aspect_ratio, rescale = 0.001,
		xlabel = "Emissions", ylabel = "Concentration"
	)

	vline!(vecfig, [c₀]; c = :black, linestyle = :dash, label = "\$c_0\$")

	eopt = optimalemissions.(cspace)
	plot!(vecfig, eopt, cspace, c = :darkred, label = "Optimal emissions")

	scatter!(vecfig, [0], [0]; c = :black, label = false)
end

# ╔═╡ 4a6c00dc-3382-4460-9ebc-40be27f22dd3
begin
	function F!(dz, z, p, t)
		m, l = p # Unpack LinearQuadratic and climate model
		(; κ, A, δ, η, S, A, cₚ) = m
		(; β₀, β₁, τ, γ, ρ, xₛ) = l
	
		x, c, λ, e = z # Unpack state
		eᵤ = (β₀ - τ) / β₁
		
		dz[1] = κ * (a(x, m) - η * x^4 + S + A * log(c / cₚ)) # Temperature 
		dz[2] = e - δ * c # Concentration 
	
		dz[3] = (ρ - κ * a′(x, m) + κ * 4η * x^3) * λ + γ * (x - xₛ) # Shadow price of temperature
		dz[4] = (ρ + δ) * (e - eᵤ) - (λ / c) * (κ * A) / β₁ # Emissions
	
		return dz
	end
	
	function DF!(D, z, p, t)
		m, l = p # Unpack a LinearQuadratic model
		(; κ, A, δ, η) = m
		(; β₀, β₁, τ, γ, ρ, xₛ) = l
	
		x, c, λ, e = z # Unpack state
	
		J = zeros(4, 4)
	
		J[1, 1] = κ * (a′(x, m) - 4η * x^3)
		J[1, 2] = κ * A / c
	
		J[2, 2] = -δ
		J[2, 4] = 1
		
		J[3, 1] = κ * (-a′′(x, m) + 12η * x^2) * λ + γ
		J[3, 3] = ρ - κ * (a′(x, m) - 4η * x^3)
	
		J[4, 2] = -(κ * A / β₁) * (λ / c^2)
		J[4, 3] = -(κ * A / β₁) * (1 / c)
		J[4, 4] = ρ + δ
	
		D .= J
	
		return J
	end

	F!, DF!
end

# ╔═╡ 75daf019-f422-4623-9418-c3992fa55897
function getequilibria(m, l; xₗ = 220, xᵤ = 320)
	(; κ, A, δ, η) = m
	(; β₀, β₁, τ, γ, ρ, xₛ) = l

	ψ(x) = climate.φ(x, m) * δ
	ω(x) = γ * (x - xₛ) / (κ * (a′(x, m) -  4η * x^3) - ρ)
	ϕ(e) = (β₁ * e) / (κ * A * δ) * (ρ + δ) * (e - (β₀ - τ) / β₁)
	
	equilibriumcond(x) = ω(x) - (ϕ ∘ ψ)(x)
	
	asymptotesω = find_zeros(x -> κ * climate.μₓ(x, m) - ρ, xₗ, xᵤ)
	
	regions = [
		(xₗ, asymptotesω[1]),
		(asymptotesω[1], asymptotesω[2]),
		(asymptotesω[2], xᵤ)
	]
	
	
	equilibria = Vector{Float64}[]
	for (l, u) ∈ regions, x ∈ find_zeros(equilibriumcond, l, u)
		e = ψ(x)
		λ = ω(x)
		c = e / δ
	
		push!(equilibria, [x, c, λ, e])
	end

	(ψ, ω, ϕ), equilibria
end

# ╔═╡ 4f9e8361-7a24-4c56-a948-057d300f3378
function computestablemanifolds(
	    F!::Function, DF!::Function,
	    steadystates::Vector{Vector{Float64}},
	    p::Vector{Any};
		alg = Tsit5(), abstol = 1.0e-10, reltol = 1.0e-10,
		h = 1e-3, tends = repeat([(10., 10.)], length(steadystates)), 
		T = 100,
		isoutofdomain = (u, p, t) -> false,
		verbose = false,
	    solverargs...)

	n = length(first(steadystates))

	function Finv!(dz, z, p, t)
		F!(dz, z, p, t)
		dz .= -dz
	end

	function DFinv!(J, z, p, t)
		DF!(J, z, p, t)
		J .= -J
	end

	odefn = ODEFunction(Finv!; jac = DFinv!)
	equil = []
		
	for (j, x̄) ∈ enumerate(steadystates)
		verbose && println("Computing manifolds for steady state: ", x̄)

		J = zeros(n, n); DF!(J, x̄, p, 0.0)
		λ, V = eigen(J)

		stabledirs = findall(λᵢ -> real(λᵢ) < 0, λ)

		manifolds = Dict()

		for i ∈ stabledirs
			vᵢ = real.(V[:, i])

			negtend = tends[j][1]
			negprob = ODEProblem(odefn, x̄ - h * vᵢ, (0.0, negtend), p)
			negsol = solve(negprob, alg; reltol = reltol, abstol = abstol, isoutofdomain = isoutofdomain, solverargs...)

			if negsol.retcode == ReturnCode.Success
				timespan = range(0.0, negtend, length = T)
				manifolds[:n] = hcat((t -> negsol(negtend - t)).(timespan)...)'
			else
				manifolds[:n] = NaN * ones(T, n)
			end


			postend = tends[j][2]
			posprob = ODEProblem(odefn, x̄ + h * vᵢ, (0.0, postend), p)
			possol = solve(posprob, alg; reltol = reltol, abstol = abstol, isoutofdomain = isoutofdomain, solverargs...)

			if possol.retcode == ReturnCode.Success
				timespan = range(0.0, postend, length = T)
				manifolds[:p] = hcat((t -> possol(postend - t)).(timespan)...)'
			else
				manifolds[:p] = NaN * ones(T, n)
			end

		end		

		push!(equil, manifolds)
	end

	return equil
end

# ╔═╡ 695e0698-2772-42df-a467-53d79b1bbe1c
l = economy.LinearQuadratic(τ = 0, xₛ = 287.3, γ = γ₀);

# ╔═╡ 331cadc1-57af-4199-b113-8f2696708916
begin
	nullfns, equilibria = getequilibria(m, l; xₗ = 100, xᵤ = 900)

	ψ, ω, ϕ = nullfns
end;

# ╔═╡ dd8b41b7-71d5-4f4f-97e4-6d5ca8906521
begin
	reltol = 1e-10
	abstol = 1e-10
	isoutofdomain = (u, p, t) -> begin
		c, x, λ, e = u
		m, l = p
	
		stateout = c ≤ m.cₚ || x < l.xₛ || λ > 0
		emissionsout = e > eᵤ
	
		return stateout && emissionsout
	end

	timehorizons = [
		(135., 110.),
		(67., 110.),
		(30., 28.)
	]

	affect!(integrator) = terminate!(integrator, SciMLBase.ReturnCode.Success)
	function instabilitycondition(u, t, integrator)
		λcond = u[3] > ω(u[1]) ? 0.0 : 1.0
		xcond = u[1] ≤ xₛ ? 0.0 : 1.0
		ccond = u[2] ≤ m.cₚ ? 0.0 : 1.0
	
		return λcond * xcond * ccond
	end

end;

# ╔═╡ 568417ca-cb93-42d3-a795-eb8bef188bec
manifolds = computestablemanifolds(
	F!, DF!, equilibria, [m, l];
	alg = Rosenbrock23(), abstol = abstol, reltol = reltol,
	isoutofdomain = isoutofdomain, 
	tends = timehorizons,
	T = 2_000, maxiters = 1e7,
	h = 1e-3
);

# ╔═╡ fb94e31a-cc84-45c9-a3c4-481972982bb5
begin
	# -- (x, c)
	xspace = range(xₛ - 2, 299; length = 2001)
	csteadystate = (x -> climate.φ(x, m)).(xspace)

	xticks = (280.5:5:295.5, (280:5:300) .- 273.5)
	aspect_ratio = (maximum(csteadystate) - minimum(csteadystate)) / (xspace[end] - xspace[1])

	xcfig = plot(
		xlims = extrema(csteadystate), ylims = extrema(xspace), 
		aspect_ratio = aspect_ratio,
		xlabel = "Carbon concentration \$c\$ in p.p.m.", 
		ylabel = "Temperature \$x\$ in °C",
		yticks = xticks
	)

	plot!(xcfig, csteadystate, xspace; c = :darkred, label = false)
	scatter!(xcfig, [c₀], [x₀]; c = :black, label = "Initial state")


	# -- (e, x)
	espace = range(-30, 45; length = 1001)

	xefig = plot(
		xlims = extrema(espace), ylims = extrema(xspace), 
		aspect_ratio = (espace[end] - espace[1]) / (xspace[end] - xspace[1]) ,
		ylabel = "Temperature \$x\$ in °C", xlabel = "Emissions \$e\$",
		yicks = xticks
		
	)
	
	hline!(xefig, [x₀]; c = :black, label = "Initial state", linestyle = :dash)
	

	# -- (x, λ)	
	λspace = range(-650, 0; length = 1001)
	xλfig = plot(
		xlims = extrema(xspace), ylims = extrema(λspace), 
		aspect_ratio = (xspace[end] - xspace[1]) / (λspace[end] - λspace[1]),
		ylabel = "Shadow price \$\\lambda_x\$", xlabel = "Temperature \$x\$",
		xticks = xticks
	)

	plot!(xλfig, xspace, ω; c = :darkred, label = nothing)
	vline!(xλfig, [x₀]; c = :black, linestyle = :dash, label = "Initial state")
	vline!(xλfig, tipping_points, c = :black, label = "Tipping points")

	# -- (c, e)
	cspace = range(100, 1000, length = 1001)
	cefig = plot(
		ylims = extrema(espace), xlims = extrema(cspace), 
		aspect_ratio = (cspace[end] - cspace[1]) / (espace[end] - espace[1]),
		xlabel = "Carbon concentration \$c\$ in p.p.m.", 
		ylabel = "Emissions \$e\$"
	)

	plot!(cefig, cspace, c -> c * m.δ; c = :darkred, label = false)
	vline!(cefig, [c₀]; c = :black, linestyle = :dash, label = "Initial state")


	# Manifolds and steady states
	figures = [xcfig, xefig, xλfig, cefig]

	colors = [:darkgreen, :darkorange, :darkblue]

	for (i, ū) ∈ enumerate(equilibria)
		x, c, λ, e = ū
		
		stablemanifolds = manifolds[i]
		
		# -- (x, c)
		for (dir, curve) ∈ stablemanifolds
			plot!(xcfig, curve[:, 2], curve[:, 1]; c = colors[i], label = nothing)
		end
		scatter!(xcfig, [c], [x]; c = colors[i], label = nothing)

		# -- (x, e)
		for (dir, curve) ∈ stablemanifolds
			plot!(xefig, curve[:, 4], curve[:, 1]; c = colors[i], label = nothing)
		end
		scatter!(xefig, [e], [x]; c = colors[i], label = nothing)
		
		# -- (x, λ)
		for (dir, curve) ∈ stablemanifolds
			plot!(xλfig, curve[:, 1], curve[:, 3]; c = colors[i], label = nothing)
		end
		scatter!(xλfig, [x], [λ]; c = colors[i], label = nothing)
		
		# -- (c, e)
		for (dir, curve) ∈ stablemanifolds
			plot!(cefig, curve[:, 2], curve[:, 4]; c = colors[i], label = nothing)
		end
		scatter!(cefig, [c], [e]; c = colors[i], label = nothing)
	end

	jointfig = plot(figures..., layout = (2, 2), size = (1200, 1200))
end

# ╔═╡ Cell order:
# ╟─24b43a3d-d657-44e4-8aca-995ee007ebb2
# ╠═83e33f5e-354d-4296-aafc-e8778d4839fb
# ╠═14b581f5-a235-47d2-baf3-065cffff6f51
# ╠═94078a8e-c9ee-4351-8d4d-c9656024934d
# ╠═fe8e8cc9-8547-4eba-8120-e4d6632fe9b8
# ╠═c5287aba-1465-46ea-bdf8-7d9cf19bf945
# ╠═e80c5fa0-dad7-11ed-22a9-c7d0f0e8f71f
# ╠═585cee5d-0ac8-49fc-b760-9200972e30a1
# ╠═67b6ee24-192b-4b19-b876-757087f26263
# ╠═09589e67-c55e-44a5-bd51-37a53e8a8585
# ╠═c1fc4206-5541-4c8e-92f3-7e072fd17a5d
# ╠═9bc05e95-a185-43b6-afe6-4707d4cefc2f
# ╠═61b868ce-b080-471d-8de8-edf807b253ce
# ╠═8d16ec1e-e16f-45d4-93a4-1cc1a18fb487
# ╠═241e7857-3178-44de-9b4e-85f0b717d131
# ╠═b78704bf-bcc4-4859-8978-fcde1a64dafb
# ╠═519620d9-c504-40ff-998e-262dbe3caeb0
# ╟─3fe60d4b-6d9d-4dbb-9f99-cbae81ef6cd3
# ╟─ff62d855-fccf-4418-b255-333832c58f29
# ╠═6fcf3aca-84a6-4d71-a498-fff91ffae7f2
# ╠═bd97a765-f358-4400-b013-3c592bd864fe
# ╟─5a2c219a-a727-44dd-917b-759667911d77
# ╠═82d4835a-2a2f-4208-ad23-9e4c9ac2cba0
# ╠═54b0c489-3c22-4206-8af3-d4cc8a372c17
# ╠═7b56773e-7d5f-411e-a5de-541ba6e7e0b0
# ╟─ad1738b0-8f7c-4c91-92a6-ad4bc096a98d
# ╠═60a24b1b-d933-45bb-8638-9167ab9dd520
# ╟─cae95c3f-698c-4287-8c68-c91186c63dc0
# ╟─eca60d2f-b7c1-4153-8a1f-268b94f66a2a
# ╟─85d4cad2-9f17-41cf-9b27-d3accb7057e8
# ╟─9f888b0c-7d8b-4213-8820-f584b3a36500
# ╟─314c6433-27ca-4aba-bd41-0e80f5c1118f
# ╟─b17c0b74-40a4-4d90-a664-7025c44fef06
# ╠═f3bd2d72-5b85-48b9-982e-782d096d5e4e
# ╠═4f9964b2-05fa-4c01-b003-7a3ca32041b4
# ╟─4a6c00dc-3382-4460-9ebc-40be27f22dd3
# ╟─75daf019-f422-4623-9418-c3992fa55897
# ╟─4f9e8361-7a24-4c56-a948-057d300f3378
# ╠═dd8b41b7-71d5-4f4f-97e4-6d5ca8906521
# ╠═695e0698-2772-42df-a467-53d79b1bbe1c
# ╠═331cadc1-57af-4199-b113-8f2696708916
# ╠═568417ca-cb93-42d3-a795-eb8bef188bec
# ╠═fb94e31a-cc84-45c9-a3c4-481972982bb5
