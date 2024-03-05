### A Pluto.jl notebook ###
# v0.19.38

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

# ╔═╡ ed817ffc-f1f4-423c-8374-975e34d449eb
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(Base.current_project())
    Pkg.instantiate()
end

# ╔═╡ 1643c6b6-e6c4-4f81-8e63-198c3ad9543e
using UnPack, JLD2, DotEnv

# ╔═╡ f3d4f91d-ebac-43cb-9789-df38f9a87a8c
using Model, Grid

# ╔═╡ bbc92008-2dfb-43e3-9e16-4c60d91a2ed1
using Random; rng = MersenneTwister(123);

# ╔═╡ e11b7094-b8d0-458c-9e4a-8ebe3b588dfc
using Plots, PlutoUI

# ╔═╡ 9a954586-c41b-44bb-917e-2262c00b958a
using Roots, Interpolations

# ╔═╡ a1f3534e-8c07-42b1-80ac-440bc016a652
using DifferentialEquations

# ╔═╡ d04d558a-c152-43a1-8668-ab3b040e6701
using DifferentialEquations: EnsembleAnalysis, EnsembleDistributed

# ╔═╡ 075e3c64-4dab-441c-88c4-51179121a6c9
using Statistics: median, mean

# ╔═╡ e8a294cb-dd67-464b-9ccb-05f4c86dfc06
using DataStructures

# ╔═╡ 93709bdd-408f-4f87-b0c8-fda34b06af57
begin
	include("../scripts/utils/plotting.jl")
	include("../scripts/utils/saving.jl")

	env = DotEnv.config()
	datapath = joinpath("..", get(env, "DATAPATH", "data/"))
end;

# ╔═╡ b29e58b6-dda0-4da9-b85d-d8d7c6472155
TableOfContents()

# ╔═╡ e74335e3-a230-4d11-8aad-3323961801aa
default(size = 500 .* (√2, 1), dpi = 180, titlefontsize = 12, linewidth = 2.5)

# ╔═╡ 5bfbf0d4-f469-4e77-b17a-329a8d192e94
ΔΛ = [0., 0.06, 0.08]; Ω = 2 .* 10 .^(-4:1/2:-1);

# ╔═╡ c4567897-ddbe-4cbe-81bf-b51e127ebf93
ω = Ω[3]

# ╔═╡ 34b780ed-ddc3-4fad-9104-6653af3d1c58
begin # Grid construction
	N = 21;
	domains = [
		Hogg().T₀ .+ (0., 9.),
		log.(Hogg().M₀ .* (1., 2.)),
		log.(Economy().Y₀ .* (0.5, 2.))
	]

	G = RegularGrid(domains, N)
end;

# ╔═╡ 6cbb7191-eb1e-4abe-8ef6-c12e07b026d5
begin
	preferences = EpsteinZin()
	calibration = load_object(joinpath(datapath, "calibration.jld2"))

	models = ModelInstance[]
	jumpmodels = ModelBenchmark[]

	for ωᵣ ∈ Ω
		economy = Economy(ωᵣ = ωᵣ)
		
		for Δλ ∈ ΔΛ
		    albedo = Albedo(λ₂ = 0.31 - Δλ)
			hogg = calibrateHogg(albedo)
			
		    model = ModelInstance(preferences, economy, hogg, albedo, calibration)		
			push!(models, model)
		end
		push!(jumpmodels, ModelBenchmark(preferences, economy, Hogg(), Jump(), calibration))
	end
end

# ╔═╡ cdc62513-a1e8-4c55-a270-761b6553d806
begin
	results = loadtotal(models, G; datapath = "../data")
	jumpresults = loadtotal(jumpmodels, G; datapath = "../data")
end;

# ╔═╡ f72b0a1a-d525-4d4b-955a-66e14d0cd764
md"# Policies"

# ╔═╡ 9bac7222-2e53-4ba6-b65e-6d1887e43f25
md"
- ``\omega:`` $(@bind ωfig Slider(Ω, show_value = true))
- ``\Delta\lambda:`` $(@bind Δλfig Slider(ΔΛ, show_value = true))
"

# ╔═╡ 5e7d4f3a-a7c2-4695-a211-448ed5909ad1
let
	k = findfirst(m -> (ωfig ≈ m.economy.ωᵣ) && (Δλfig ≈ m.albedo.λ₁ - m.albedo.λ₂), models)
	mₖ = models[k]
	timesteps, V, policy = results[k]

	# Policies

	χₖ = first.(policy[:, :, :, 1])
	αₖ = last.(policy[:, :, :, 1])
	
	αfig = plotsection(αₖ, log(mₖ.economy.Y₀), G; zdim = 3, c = :Blues, linewidth = 0, levels = 30, title = "\$\\alpha\$")
	χfig = plotsection(χₖ, log(mₖ.economy.Y₀), G; zdim = 3, c = :Blues, clims = (0., 1), linewidth = 0, levels = 30, title = "\$\\chi\$")

	Tspace = range(G.domains[1]...; length = 101)
	nullcline = [Model.mstable(T, mₖ.hogg, Albedo(λ₂ = Albedo().λ₁ - Δλfig)) for T ∈ Tspace]

	plot!(αfig, nullcline, Tspace; c = :black, linewidth = 3, linestyle = :dash, label = false)
	plot!(χfig, nullcline, Tspace; c = :black, linewidth = 3, linestyle = :dash, label = false)


	plot(αfig, χfig)
	
end

# ╔═╡ 030fb1b2-8c01-4a0a-be11-94dd77b5379a
let
	k = findfirst(m -> (ωfig ≈ m.economy.ωᵣ) && (Δλfig ≈ m.albedo.λ₁ - m.albedo.λ₂), models)
	mₖ = models[k]
	timesteps, V, policy = results[k]

	Vfig = plotsection(10e16 * V[:, :, :, 1], log(mₖ.economy.Y₀), G; zdim = 3, surf = true, xflip = true, camera = (30, 45))
end

# ╔═╡ 6fe67c9b-fe20-42e4-b817-b31dad586e55
md"# Backward Simulation"

# ╔═╡ fc2c9720-3607-4ee2-a48c-f8e22d4404bd
md"## Constructing interpolations"

# ╔═╡ d2c83cdf-002a-47dc-81f9-22b76f183587
begin
	ΔT, Δm, Δy = G.domains
	spacenodes = ntuple(i -> range(G.domains[i]...; length = N), 3)

	resultsmap = OrderedDict()
	jumpresultsmap = OrderedDict()

	for ωᵣ ∈ Ω
		j = findfirst(m -> ωᵣ ≈ m.economy.ωᵣ, jumpmodels)

		res = jumpresults[j]	
		ts, V, policy = res
			
		nodes = (spacenodes..., ts)
		χitp = linear_interpolation(nodes, first.(policy); extrapolation_bc = Flat())
		αitp = linear_interpolation(nodes, last.(policy); extrapolation_bc = Flat())
		Vitp = linear_interpolation(nodes, 10e16 * V; extrapolation_bc = Flat())

		
		jumpresultsmap[ωᵣ] = (χitp, αitp, Vitp, jumpmodels[j])
		
		for Δλ ∈ ΔΛ
			k = findfirst(m -> ωᵣ ≈ m.economy.ωᵣ && Δλ ≈ m.albedo.λ₁ - m.albedo.λ₂, models)

			res = results[k]
			model = models[k]
					
			ts, V, policy = res
				
			nodes = (spacenodes..., ts)
			χitp = linear_interpolation(nodes, first.(policy); extrapolation_bc = Flat())
			αitp = linear_interpolation(nodes, last.(policy); extrapolation_bc = Flat())
			Vitp = linear_interpolation(nodes, V; extrapolation_bc = Flat())

			
			resultsmap[(ωᵣ, Δλ)] = (χitp, αitp, Vitp, model)
		end
	end
end;

# ╔═╡ 5c553323-3611-4614-8de3-86ea5cf8eea0
Eᵇ = linear_interpolation(calibration.years .- 2020, calibration.emissions, extrapolation_bc = Line());

# ╔═╡ 31d32b62-4329-464e-8354-1c2875fe5801
md"## Simulation"

# ╔═╡ d62b65f5-220e-45c6-a434-ac392b72ab4a
function F!(dx, x, p::Tuple{ModelInstance, Any, Any}, t)	
	model, χitp, αitp = p
	
	T, m, y = x
	
	χ = χitp(T, m, y, t)
	α = αitp(T, m, y, t)
	
	dx[1] = μ(T, m, model.hogg, model.albedo) / model.hogg.ϵ
	dx[2] = γ(t, model.economy, model.calibration) - α
	dx[3] = b(t, Point(x), Policy(χ, α), model)

	return
end;

# ╔═╡ 7823bda7-5ab8-42f7-bf1c-292dbfecf178
function G!(dx, x, p, t)
	model = first(p)
	
	dx[1] = model.hogg.σₜ / model.hogg.ϵ
	dx[2] = 0.
	dx[3] = model.economy.σₖ
	
	return
end;

# ╔═╡ f6cb362d-240c-42fe-bbf7-7a26fdb189ae
begin
	x₀ = [Hogg().T₀, log(Hogg().M₀), log(Economy().Y₀)]
	X₀ = Point(x₀)
end;

# ╔═╡ b1a04663-c0be-45d2-9846-46ca8485219c
md"# Comparison with Jump"

# ╔═╡ 96df56f0-6f1d-4a30-b26a-6a53307e79ca
function F!(dx, x, p::Tuple{ModelBenchmark, Any, Any}, t)	
	model, χitp, αitp = p
	
	T, m, y = x
	
	χ = χitp(T, m, y, t)
	α = αitp(T, m, y, t)
	
	dx[1] = μ(T, m, model.hogg) / model.hogg.ϵ
	dx[2] = γ(t, model.economy, model.calibration) - α
	dx[3] = b(t, Point(x), Policy(χ, α), model)

	return
end;

# ╔═╡ c3c2cfc9-e7f4-495b-bcc6-51227be2c6b5
begin
	tspan = (0., Economy().t₁)

	problems = OrderedDict{Float64, SDEProblem}()
	fn = SDEFunction(F!, G!)

	for Δλ ∈ ΔΛ
		χitp, αitp, _, model = resultsmap[(ω, Δλ)]
		parameters = (model, χitp, αitp)
		
		problems[Δλ] = SDEProblem(fn, x₀, tspan, parameters)
	end	

	# Use the model for Δλ = 8% with the policy from Δλ = 0%
	χitpnaive, αitpnaive, _, _ = resultsmap[(ω, 0.)]
	modelnaive = last(resultsmap[(ω, 0.08)])

	problems[-1.] = SDEProblem(fn, x₀, tspan, (modelnaive, χitpnaive, αitpnaive))
end;

# ╔═╡ 28c6fe28-bd42-4aba-b403-b2b0145a8e37
solutions = OrderedDict(key => solve(EnsembleProblem(prob), EnsembleDistributed(); trajectories = 150) for (key, prob) ∈ problems);

# ╔═╡ 1a19b769-68e2-411b-afe0-6bd2a7fb87a3
begin	
	colors = Dict(keys(solutions) .=> [:darkgreen, :darkblue, :darkred, :black])


	yearticks = 2020 .+ (0:10:80)
	xticks = (0:10:80, yearticks)
		
	Tticks = Hogg().Tᵖ .+ (1:3)
	yticks = (Tticks, 1:3)
	
	
	Tfig = hline(; ylabel = "\$T - T^p\$", legendtitle = "\$\\Delta \\lambda\$", ylims = extrema(Tticks), yticks = yticks, xticks = xticks)
	
	Mfig = plot(; ylabel = "\$M\$", xticks)
	Yfig = plot(; ylabel = "\$Y\$", xlabel = "year", xticks)

	timespan = range(tspan...; length = 301)

	for (Δλ, solution) ∈ solutions
		median = [Point(EnsembleAnalysis.timepoint_median(solution, tᵢ)) for tᵢ ∈ timespan]
		
		color = colors[Δλ]
		label = Δλ ≥ 0 ? Δλ : "Naive"

		plot!(Tfig, timespan, [x.T for x ∈ median]; c = color, label = label, linewidth = 3, alpha = 0.8)
		plot!(Mfig, timespan, [exp(x.m) for x ∈ median]; c = color, label = false, linewidth = 3, alpha = 0.8)
		plot!(Yfig, timespan, [exp(x.y) for x ∈ median]; c = color, label = false, linewidth = 3, alpha = 0.8)
	end

	plot(Tfig, Mfig, Yfig, sharex = true, layout = (3, 1), link = :x, size = 500 .* (√2, 2.5), margins = 5Plots.mm)
end

# ╔═╡ 468d5245-ff79-44df-938f-72ad656fe385
begin	
	Tdomain, mdomain, _ = G.domains
	Tspace = range(Tdomain...; length = 101)
	mspace = range(mdomain[1], 6.4; length = 101)

	coarsetimespan = range(first(timespan), last(timespan), step = 10)

	trajfig = plot(xlims = (minimum(mspace) - 0.05, maximum(mspace) + 0.1), ylims = Hogg().Tᵖ .+ (1., 7.))

	for (Δλ, sol) ∈ solutions

		if Δλ < 0
			continue
		end

		model = resultsmap[(first(Ω), Δλ)] |> last
		color = colors[Δλ]
	
		nullcline = [Model.mstable(T, model.hogg, model.albedo) for T ∈ Tspace]
	
		trajectory = [Point(EnsembleAnalysis.timepoint_median(sol, tᵢ)) for tᵢ ∈ coarsetimespan]
		mtraj = [x.m for x ∈ trajectory]
		Ttraj = [x.T for x ∈ trajectory]
	
		plot!(trajfig, mtraj, Ttraj; c = color, label = Δλ, linewidth = 3, marker = :o)
		plot!(trajfig, nullcline, Tspace; c = color, linestyle = :dash, label = false, linewidth = 3)

	end

	trajfig
end

# ╔═╡ 546a3db3-e485-411d-915b-bd5ebe870779
begin
	χitp, αitp, _, model = jumpresultsmap[ω]
	parameters = (model, χitp, αitp)
			
	jumpproblem = SDEProblem(fn, x₀, tspan, parameters)
	jumpsolution = solve(EnsembleProblem(jumpproblem), EnsembleDistributed(), trajectories = 50)
end;

# ╔═╡ a6d237d3-e150-41f2-8861-b94349a279ba
function scc(solution, Vitp)
	c = Matrix{Float64}(undef, length(timespan), length(solution))

	for (j, sim) ∈ enumerate(solution), (i, tᵢ) ∈ enumerate(timespan)
		T, m, y = sim(tᵢ)
		M, Y = exp(m), exp(y)
				
		_, ∂V∂m, ∂V∂y = Interpolations.gradient(Vitp, T, m, y, tᵢ)
		
		∂V∂E = ∂V∂m / (M *  Model.Gtonoverppm)
		∂V∂Y = ∂V∂y / Y

		c[i, j] = -∂V∂E / ∂V∂Y
	end

	return c
end;

# ╔═╡ cd111495-5032-4557-a681-22a880fe4fbd
function emissionpath(solution, model, αitp)
	E = Matrix{Float64}(undef, length(timespan), length(solution))

	for (j, sim) ∈ enumerate(solution), (i, tᵢ) ∈ enumerate(timespan)
		T, m, y = sim(tᵢ)
		αₜ = αitp(T, m, y, tᵢ)
		
		M = exp(m)
		Eₜ = (M / Model.Gtonoverppm) * (γ(tᵢ, model.economy, model.calibration) - αₜ)

		
		E[i, j] = Eₜ
	end

	return E
end;

# ╔═╡ 0ef87388-2261-4af8-9857-b30623e8f008
function consumptionpath(solution, model, χitp)
	C = Matrix{Float64}(undef, length(timespan), length(solution))

	for (j, sim) ∈ enumerate(solution), (i, tᵢ) ∈ enumerate(timespan)
		T, m, y = sim(tᵢ)
		χᵢ = χitp(T, m, y, tᵢ)
		
		C[i, j] = exp(y) * χᵢ
	end

	return C
end;

# ╔═╡ 85b19b82-8b96-4665-9e3f-256f40e91b49
function variablepath(solution, model)
	X = Matrix{Point}(undef, length(timespan), length(solution))

	for (j, sim) ∈ enumerate(solution), (i, tᵢ) ∈ enumerate(timespan)
		T, m, y = sim(tᵢ)
		X[i, j] = Point(T, m, y)
	end

	return X
end;

# ╔═╡ 6c3a54f3-6739-4785-9b27-f0468e5c3030
let
	Efig = plot(; ylabel = "Net emissions", xlabel = "year", xticks, dpi = 180)
	
	# Albedo
	_, αitp, _, model = resultsmap[(ω, 0.08)]
	solution = solutions[0.08]
	emissions = emissionpath(solution, model, αitp)

	plot!(timespan, median(emissions, dims = 2); c = :darkred, label = "Albedo 8%")

	for E ∈ eachcol(emissions)
		plot!(timespan, E; label = nothing, alpha = 0.05, c = :darkred)
	end

	# Albedo
	_, αitp, _, model = resultsmap[(ω, 0.06)]
	solution = solutions[0.06]
	emissions = emissionpath(solution, model, αitp)

	plot!(timespan, median(emissions, dims = 2); c = :darkorange, label = "Albedo 6%")

	for E ∈ eachcol(emissions)
		plot!(timespan, E; label = nothing, alpha = 0.05, c = :darkorange)
	end

	
	# Jump
	_, αitp, _, jumpmodel = jumpresultsmap[ω]
	emissions = emissionpath(jumpsolution, jumpmodel, αitp)

	plot!(timespan, median(emissions, dims = 2); c = :darkblue, label = "Stochastic tipping")

	for E ∈ eachcol(emissions)
		plot!(timespan, E; label = nothing, alpha = 0.05, c = :darkblue)
	end
			

	savefig("../plots/optemissions.png")
	Efig
end

# ╔═╡ c0af239e-5494-4372-8549-a312074bd91f
let
	Mfig = plot(; ylabel = "Carbon concentration \$M\$ [p.p.m.]", xlabel = "year", xticks, dpi = 180)
	
	# Albedo 8 %
	model = last(resultsmap[(ω, 0.08)])
	solution = solutions[0.08]

	X = variablepath(solution, model)
	M = [exp(x.m) for x ∈ X]

	plot!(timespan, median(M, dims = 2); c = :darkred, label = "Albedo 8%")

	for Mᵢ ∈ eachcol(M)
		plot!(timespan, Mᵢ; label = nothing, alpha = 0.05, c = :darkred)
	end

	# Albedo 6 %
	model = last(resultsmap[(ω, 0.06)])
	solution = solutions[0.06]

	X = variablepath(solution, model)
	M = [exp(x.m) for x ∈ X]

	plot!(timespan, median(M, dims = 2); c = :darkorange, label = "Albedo 6%")

	for Mᵢ ∈ eachcol(M)
		plot!(timespan, Mᵢ; label = nothing, alpha = 0.05, c = :darkorange)
	end

	
	# Jump
	jumpmodel = jumpresultsmap[ω] |> last
	X = variablepath(jumpsolution, jumpmodel)
	M = [exp(x.m) for x ∈ X]

	plot!(timespan, median(M, dims = 2); c = :darkblue, label = "Stochastic tipping")

	for Mᵢ ∈ eachcol(M)
		plot!(timespan, Mᵢ; label = nothing, alpha = 0.05, c = :darkblue)
	end
			

	savefig("../plots/optcarbon.png")
	Mfig
end

# ╔═╡ d38bf362-f2c1-483a-88de-8a731db29af8
let
	Yfig = plot(; ylabel = "GDP / Consumption [Trillion US\$]", xlabel = "year", xticks, dpi = 180, ylims = (0, Inf))
	
	# Albedo 8 %
	χitp, _, _, model = resultsmap[(ω, 0.08)]
	solution = solutions[0.08]

	X = variablepath(solution, model)
	Y = [exp(x.y) for x ∈ X]
	C = consumptionpath(solution, model, χitp)

	plot!(timespan, median(Y, dims = 2); c = :darkred, label = "Albedo 8%")
	plot!(timespan, median(C, dims = 2); c = :darkred, label = nothing, linestyle = :dash)

	# Albedo 6 %
	χitp, _, _, model = resultsmap[(ω, 0.06)]
	solution = solutions[0.06]

	X = variablepath(solution, model)
	Y = [exp(x.y) for x ∈ X]
	C = consumptionpath(solution, model, χitp)

	plot!(timespan, median(Y, dims = 2); c = :darkorange, label = "Albedo 6%")
	plot!(timespan, median(C, dims = 2); c = :darkorange, label = nothing, linestyle = :dash)

	
	# Jump
	jumpχitp, _, _, jumpmodel = jumpresultsmap[ω]
	X = variablepath(jumpsolution, jumpmodel)
	Y = [exp(x.y) for x ∈ X]
	C = consumptionpath(jumpsolution, jumpmodel, jumpχitp)

	plot!(timespan, median(Y, dims = 2); c = :darkblue, label = "Stochastic tipping")
	plot!(timespan, median(C, dims = 2); c = :darkblue, label = nothing, linestyle = :dash)

			

	savefig("../plots/optgdp.png")
	Yfig
end

# ╔═╡ aea5f8ac-349b-4eda-a2bf-2bfaf4d08e7e
let
	sccfig = plot(; ylabel = "GDP / Consumption [Trillion US\$]", xlabel = "year", xticks, dpi = 180, ylims = (0, Inf))
	
	# Albedo 8 %
	_, _, Vitp, model = resultsmap[(ω, 0.08)]
	solution = solutions[0.08]

	sccpath = scc(solution, Vitp)

	plot!(timespan, median(sccpath, dims = 2); c = :darkred, label = "Albedo 8%")

	# Albedo 6 %
	_, _, Vitp, model = resultsmap[(ω, 0.06)]
	solution = solutions[0.06]

	sccpath = scc(solution, Vitp)

	plot!(timespan, median(sccpath, dims = 2); c = :darkorange, label = "Albedo 6%")

	
	# Jump
	_, _, Vitp, jumpmodel = jumpresultsmap[ω]
	sccpath = scc(jumpsolution, Vitp)

	plot!(timespan, median(sccpath, dims = 2); c = :darkblue, label = "Stochastic tipping")

			
	sccfig
end

# ╔═╡ Cell order:
# ╠═ed817ffc-f1f4-423c-8374-975e34d449eb
# ╠═1643c6b6-e6c4-4f81-8e63-198c3ad9543e
# ╠═f3d4f91d-ebac-43cb-9789-df38f9a87a8c
# ╠═bbc92008-2dfb-43e3-9e16-4c60d91a2ed1
# ╠═e11b7094-b8d0-458c-9e4a-8ebe3b588dfc
# ╠═9a954586-c41b-44bb-917e-2262c00b958a
# ╠═b29e58b6-dda0-4da9-b85d-d8d7c6472155
# ╠═e74335e3-a230-4d11-8aad-3323961801aa
# ╠═93709bdd-408f-4f87-b0c8-fda34b06af57
# ╠═5bfbf0d4-f469-4e77-b17a-329a8d192e94
# ╠═c4567897-ddbe-4cbe-81bf-b51e127ebf93
# ╠═34b780ed-ddc3-4fad-9104-6653af3d1c58
# ╠═6cbb7191-eb1e-4abe-8ef6-c12e07b026d5
# ╠═cdc62513-a1e8-4c55-a270-761b6553d806
# ╟─f72b0a1a-d525-4d4b-955a-66e14d0cd764
# ╟─9bac7222-2e53-4ba6-b65e-6d1887e43f25
# ╟─5e7d4f3a-a7c2-4695-a211-448ed5909ad1
# ╠═030fb1b2-8c01-4a0a-be11-94dd77b5379a
# ╟─6fe67c9b-fe20-42e4-b817-b31dad586e55
# ╠═a1f3534e-8c07-42b1-80ac-440bc016a652
# ╠═d04d558a-c152-43a1-8668-ab3b040e6701
# ╠═075e3c64-4dab-441c-88c4-51179121a6c9
# ╠═e8a294cb-dd67-464b-9ccb-05f4c86dfc06
# ╟─fc2c9720-3607-4ee2-a48c-f8e22d4404bd
# ╠═d2c83cdf-002a-47dc-81f9-22b76f183587
# ╠═5c553323-3611-4614-8de3-86ea5cf8eea0
# ╟─31d32b62-4329-464e-8354-1c2875fe5801
# ╠═d62b65f5-220e-45c6-a434-ac392b72ab4a
# ╠═7823bda7-5ab8-42f7-bf1c-292dbfecf178
# ╠═f6cb362d-240c-42fe-bbf7-7a26fdb189ae
# ╠═c3c2cfc9-e7f4-495b-bcc6-51227be2c6b5
# ╠═28c6fe28-bd42-4aba-b403-b2b0145a8e37
# ╠═1a19b769-68e2-411b-afe0-6bd2a7fb87a3
# ╟─468d5245-ff79-44df-938f-72ad656fe385
# ╟─b1a04663-c0be-45d2-9846-46ca8485219c
# ╠═96df56f0-6f1d-4a30-b26a-6a53307e79ca
# ╠═546a3db3-e485-411d-915b-bd5ebe870779
# ╠═a6d237d3-e150-41f2-8861-b94349a279ba
# ╠═cd111495-5032-4557-a681-22a880fe4fbd
# ╠═0ef87388-2261-4af8-9857-b30623e8f008
# ╠═85b19b82-8b96-4665-9e3f-256f40e91b49
# ╟─6c3a54f3-6739-4785-9b27-f0468e5c3030
# ╟─c0af239e-5494-4372-8549-a312074bd91f
# ╟─d38bf362-f2c1-483a-88de-8a731db29af8
# ╠═aea5f8ac-349b-4eda-a2bf-2bfaf4d08e7e
