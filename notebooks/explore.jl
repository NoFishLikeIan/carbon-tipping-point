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
ΔΛ = [0., 0.06, 0.08]; Ω = 2 * 10. .^(-4:1:-1);

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
	

	for Δλ ∈ ΔΛ, ωᵣ ∈ Ω
		economy = Economy(ωᵣ = ωᵣ)
	    albedo = Albedo(λ₂ = 0.31 - Δλ)
		hogg = calibrateHogg(albedo)
	    model = ModelInstance(preferences, economy, hogg, albedo, calibration)

		push!(models, model)
	end
end

# ╔═╡ cdc62513-a1e8-4c55-a270-761b6553d806
results = loadtotal(models, G; datapath = "../data");

# ╔═╡ f72b0a1a-d525-4d4b-955a-66e14d0cd764
md"## Initial exploration of results"

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

	χₖ = first.(policy[:, :, :, 1])
	αₖ = last.(policy[:, :, :, 1])
	
	αfig = plotsection(αₖ, log(mₖ.economy.Y₀), G; zdim = 3, c = :Blues, linewidth = 0, clims = (0., 0.0225), levels = 30, title = "\$\\alpha\$")
	χfig = plotsection(χₖ, log(mₖ.economy.Y₀), G; zdim = 3, c = :Blues, clims = (0., 1), linewidth = 0, levels = 30, title = "\$\\chi\$")

	Tspace = range(G.domains[1]...; length = 101)
	nullcline = [Model.mstable(T, mₖ.hogg, Albedo(λ₂ = Albedo().λ₁ - Δλfig)) for T ∈ Tspace]

	plot!(αfig, nullcline, Tspace; c = :black, linewidth = 3, linestyle = :dash, label = false)
	plot!(χfig, nullcline, Tspace; c = :black, linewidth = 3, linestyle = :dash, label = false)

	plot(αfig, χfig)
	
end

# ╔═╡ 6fe67c9b-fe20-42e4-b817-b31dad586e55
md"# Backward Simulation"

# ╔═╡ fc2c9720-3607-4ee2-a48c-f8e22d4404bd
md"## Constructing interpolations"

# ╔═╡ d2c83cdf-002a-47dc-81f9-22b76f183587
begin
	ΔT, Δm, Δy = G.domains
	spacenodes = ntuple(i -> range(G.domains[i]...; length = N), 3)

	interpolations = []

	for (k, res) in enumerate(results)
		ts, V, policy = res
			
		nodes = (spacenodes..., ts)
		χitp = linear_interpolation(nodes, first.(policy); extrapolation_bc = Flat())
		αitp = linear_interpolation(nodes, last.(policy); extrapolation_bc = Flat())
		Vitp = linear_interpolation(nodes, V; extrapolation_bc = Flat())

		push!(interpolations, (χitp, αitp, Vitp))
	end
end;

# ╔═╡ 5c553323-3611-4614-8de3-86ea5cf8eea0
Eᵇ = linear_interpolation(calibration.years .- 2020, calibration.emissions, extrapolation_bc = Line());

# ╔═╡ 31d32b62-4329-464e-8354-1c2875fe5801
md"## Simulation"

# ╔═╡ d62b65f5-220e-45c6-a434-ac392b72ab4a
function F!(dx, x, p, t)	
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

# ╔═╡ 9ecd7282-3394-4b4f-8af3-cb1a01032873
md"``\omega:`` $(@bind ωᵣ Slider(Ω, show_value = true))"

# ╔═╡ c3c2cfc9-e7f4-495b-bcc6-51227be2c6b5
begin
	tspan = (0., Economy().t₁)

	problems = SDEProblem[]
	fn = SDEFunction(F!, G!)

	for Δλ ∈ ΔΛ
		k = findfirst(m -> ωᵣ ≈ m.economy.ωᵣ && Δλ ≈ m.albedo.λ₁ - m.albedo.λ₂, models)
		model = models[k]
		itps = interpolations[k]
		
		parameters = (model, itps[1], itps[2])
		
		push!(problems, SDEProblem(fn, x₀, tspan, parameters))
	end	
end;

# ╔═╡ 28c6fe28-bd42-4aba-b403-b2b0145a8e37
solutions = [solve(EnsembleProblem(prob), EnsembleDistributed(); trajectories = 50) for prob ∈ problems];

# ╔═╡ 1a19b769-68e2-411b-afe0-6bd2a7fb87a3
begin
	colors = [:darkblue, :darkred, :darkgreen, :darkorange]

	yearticks = 2020 .+ (0:10:80)
	xticks = (0:10:80, yearticks)
		
	Tticks = Hogg().Tᵖ .+ (1:3)
	yticks = (Tticks, 1:3)
	
	
	Tfig = hline(; ylabel = "\$T - T^p\$", legendtitle = "\$\\Delta \\lambda\$", ylims = extrema(Tticks), yticks = yticks, xticks = xticks)
	
	Mfig = plot(; ylabel = "\$M\$", xticks)
	Yfig = plot(; ylabel = "\$Y\$", xlabel = "year", xticks)

	timespan = range(tspan...; length = 101)

	for (i, solution) ∈ enumerate(solutions)
		median = [Point(EnsembleAnalysis.timepoint_median(solution, tᵢ)) for tᵢ ∈ timespan]

		plot!(Tfig, timespan, [x.T for x ∈ median]; c = colors[i], label = ΔΛ[i], linewidth = 3)
		plot!(Mfig, timespan, [exp(x.m) for x ∈ median]; c = colors[i], label = false, linewidth = 3)
		plot!(Yfig, timespan, [exp(x.y) for x ∈ median]; c = colors[i], label = false, linewidth = 3)
		
		for sim in solution
			data = Point.(sim.(timespan))
			plot!(Tfig, timespan, [ x.T for x ∈ data ]; label = false, alpha = 0.05, c = colors[i])
			plot!(Yfig, timespan, [exp(x.y) for x ∈ data]; c = colors[i], label = false, alpha = 0.05)
		end
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

	for (k, Δλ) ∈ enumerate(ΔΛ)
		j = findfirst(m -> ωᵣ ≈ m.economy.ωᵣ && Δλ ≈ m.albedo.λ₁ - m.albedo.λ₂, models)
		model = models[j]
		
		nullcline = [Model.mstable(T, model.hogg, model.albedo) for T ∈ Tspace]
	
		trajectory = [Point(EnsembleAnalysis.timepoint_median(solutions[k], tᵢ)) for tᵢ ∈ coarsetimespan]
		mtraj = [x.m for x ∈ trajectory]
		Ttraj = [x.T for x ∈ trajectory]
	
		plot!(trajfig, mtraj, Ttraj; c = colors[k], label = Δλ, linewidth = 3, marker = :o)
		plot!(trajfig, nullcline, Tspace; c = colors[k], linestyle = :dash, label = false, linewidth = 3)

	end

	trajfig
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
# ╠═34b780ed-ddc3-4fad-9104-6653af3d1c58
# ╠═6cbb7191-eb1e-4abe-8ef6-c12e07b026d5
# ╠═cdc62513-a1e8-4c55-a270-761b6553d806
# ╟─f72b0a1a-d525-4d4b-955a-66e14d0cd764
# ╟─9bac7222-2e53-4ba6-b65e-6d1887e43f25
# ╟─5e7d4f3a-a7c2-4695-a211-448ed5909ad1
# ╟─6fe67c9b-fe20-42e4-b817-b31dad586e55
# ╠═a1f3534e-8c07-42b1-80ac-440bc016a652
# ╠═d04d558a-c152-43a1-8668-ab3b040e6701
# ╠═075e3c64-4dab-441c-88c4-51179121a6c9
# ╟─fc2c9720-3607-4ee2-a48c-f8e22d4404bd
# ╠═d2c83cdf-002a-47dc-81f9-22b76f183587
# ╠═5c553323-3611-4614-8de3-86ea5cf8eea0
# ╟─31d32b62-4329-464e-8354-1c2875fe5801
# ╠═d62b65f5-220e-45c6-a434-ac392b72ab4a
# ╠═7823bda7-5ab8-42f7-bf1c-292dbfecf178
# ╠═f6cb362d-240c-42fe-bbf7-7a26fdb189ae
# ╠═c3c2cfc9-e7f4-495b-bcc6-51227be2c6b5
# ╠═28c6fe28-bd42-4aba-b403-b2b0145a8e37
# ╟─9ecd7282-3394-4b4f-8af3-cb1a01032873
# ╟─1a19b769-68e2-411b-afe0-6bd2a7fb87a3
# ╠═468d5245-ff79-44df-938f-72ad656fe385
