### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

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
default(size = 500 .* (√2, 1), dpi = 180, titlefontsize = 12)

# ╔═╡ 0985734e-0c1e-4e4c-8ec5-191be880a72f
begin
	loaddata(N::Int64, Δλ::Real, p::Preferences) = loaddata(N, [Δλ], p)
	function loaddata(N::Int64, ΔΛ::AbstractVector{<:Real}, p::Preferences)
		termpath = joinpath(datapath, "terminal")
		simpath = joinpath(datapath, "total")
		
		G = load(joinpath(termpath, filename(N, first(ΔΛ), p)), "G")
		model = load(joinpath(termpath, filename(N, first(ΔΛ), p)), "model")
		@unpack economy, calibration = model
	
		timesteps = range(0, economy.τ; step = 0.25)
		V = Array{Float64}(undef, N, N, N, length(ΔΛ), length(timesteps))
		policy = similar(V, Policy)
	
		ᾱ = γ(economy.τ, economy, calibration)
	
		for (k, Δλ) ∈ enumerate(ΔΛ)
			name = filename(N, Δλ, p)
			V[:, :, :, k, end] .= load(joinpath(termpath, name), "V̄")
			policy[:, :, :, k, end] .= [Policy(χ, ᾱ) for χ ∈ load(joinpath(termpath, name), "policy")]
	
			file = jldopen(joinpath(simpath, name), "r")
	
			for (j, tᵢ) ∈ enumerate(timesteps[1:(end - 1)])
				idx = size(V, 5) - j
				V[:, :, :, k, idx] .= file[string(tᵢ)]["V"]
				policy[:, :, :, k, idx] .= file[string(tᵢ)]["policy"]
			end
	
			close(file)
		end
	
		return timesteps, V, policy, model, G
	end
end;

# ╔═╡ cdc62513-a1e8-4c55-a270-761b6553d806
begin
	ΔΛ = [0., 0.08]
	p = CRRA(θ = 1.45)
	N = 31
	
	t, V, policy, model, G = loaddata(N, ΔΛ, p)
end;

# ╔═╡ acc678b2-01bd-4504-ad88-f3f926cd9518
begin
	@unpack hogg, economy, calibration, albedo = model
	X₀ = Point([hogg.T₀, log(hogg.M₀), log(economy.Y₀)])
end;

# ╔═╡ 5e7d4f3a-a7c2-4695-a211-448ed5909ad1
plotsection(V[:, :, :, end, end - 1], log(economy.Y₀), G; zdim = 3, surf = true, xflip = true)

# ╔═╡ 6fe67c9b-fe20-42e4-b817-b31dad586e55
md"# Backward Simulation"

# ╔═╡ fc2c9720-3607-4ee2-a48c-f8e22d4404bd
md"## Constructing interpolations"

# ╔═╡ d2c83cdf-002a-47dc-81f9-22b76f183587
begin
	ΔT, Δm, Δy = G.domains

	nodes = (
		range(extrema(ΔT)...; length = N),
		range(extrema(Δm)...; length = N),
		range(extrema(Δy)...; length = N),
		ΔΛ, t
	)
end;

# ╔═╡ 08a3333b-bac8-426e-a888-9bf5269c1869
begin
	χitp = linear_interpolation(nodes, first.(policy); extrapolation_bc = Flat())
	αitp = linear_interpolation(nodes, last.(policy); extrapolation_bc = Flat())
end;

# ╔═╡ 31d32b62-4329-464e-8354-1c2875fe5801
md"## Simulation"

# ╔═╡ 5c553323-3611-4614-8de3-86ea5cf8eea0
Eᵇ = linear_interpolation(calibration.years .- 2020, calibration.emissions, extrapolation_bc = Line());

# ╔═╡ d62b65f5-220e-45c6-a434-ac392b72ab4a
function F!(dx, x, p, t)	
	Δλ = first(p)
	
	T, m, y = x
	
	χ = χitp(T, m, y, Δλ, t)
	α = αitp(T, m, y, Δλ, t)
	
	dx[1] = μ(T, m, hogg, Albedo(λ₂ = albedo.λ₁ - Δλ)) / hogg.ϵ
	dx[2] = γ(t, economy, calibration) - α
	dx[3] = b(t, Point(x), Policy(χ, α), model)

	return
end;

# ╔═╡ 7823bda7-5ab8-42f7-bf1c-292dbfecf178
function G!(dx, x, p, t)
	dx[1] = hogg.σₜ / hogg.ϵ
	dx[2] = 0.
	dx[3] = economy.σₖ
	
	return
end;

# ╔═╡ c3c2cfc9-e7f4-495b-bcc6-51227be2c6b5
begin
	tspan = (0., 40.)
	x₀ = [X₀.T, X₀.m, X₀.y]
	problems = [SDEProblem(SDEFunction(F!, G!), x₀, tspan, (Δλ, )) for Δλ ∈ ΔΛ]
end;

# ╔═╡ 28c6fe28-bd42-4aba-b403-b2b0145a8e37
begin
	solutions = [solve(EnsembleProblem(prob), EnsembleDistributed(); trajectories = 100) for prob ∈ problems]
end;

# ╔═╡ 1a19b769-68e2-411b-afe0-6bd2a7fb87a3
begin
	colors = [:darkblue, :darkred, :darkgreen]

	yearticks = 2020 .+ (0:10:80)
	xticks = (0:10:80, yearticks)
		
	Tticks = hogg.Tᵖ .+ (1:2.5)
	yticks = (Tticks, 1:2.5)
	
	
	Tfig = plot(; ylabel = "\$T - T^p\$", legendtitle = "\$\\Delta \\lambda\$", yticks, ylims = extrema(Tticks), xticks)
	
	Mfig = plot(; ylabel = "\$M\$", xlabel = "year", xticks)

	timespan = range(tspan...; length = 101)

	for (i, solution) ∈ enumerate(solutions)
		median = [Point(EnsembleAnalysis.timepoint_median(solution, tᵢ)) for tᵢ ∈ timespan]

		plot!(Tfig, timespan, [x.T for x ∈ median]; c = colors[i], label = ΔΛ[i], linewidth = 3)
		plot!(Mfig, timespan, [exp(x.m) for x ∈ median]; c = colors[i], label = false, linewidth = 3)
		
		for sim in solution
			data = Point.(sim.(timespan))
			plot!(Tfig, timespan, [ x.T for x ∈ data ]; label = false, alpha = 0.05, c = colors[i])
		end
	end

	plot(Tfig, Mfig, sharex = true, layout = (2, 1), link = :x, size = 500 .* (√2, 1.5), margins = 5Plots.mm)
end

# ╔═╡ 43bc8d15-40d5-457c-84f9-57826cb4139f
begin
	efig = plot(ylabel = "\$E\$")
	
	for (i, solution) ∈ enumerate(solutions)
		Δλ = ΔΛ[i]

		T, M = length(t), length(solution)
		emissions = Matrix{Float64}(undef, T, M)

		for (i, tᵢ) ∈ enumerate(t)
			data = Point.(solution(tᵢ))

			for (j, x) ∈ enumerate(data)
				M = exp(x.m)
				E = (1 - Model.ε(tᵢ, exp(x.m), αitp(x..., Δλ, tᵢ), model)) * Eᵇ(tᵢ)
				
				emissions[i, j] = γ(tᵢ, economy, calibration) - αitp(x..., Δλ, tᵢ)
			end
		end


		
		plot!(efig, median(emissions, dims = 2); c = colors[i], linewidth = 2, label = Δλ)
	end
	

	efig
end

# ╔═╡ aa6cdd38-8733-425f-ac39-29d031f7c269


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
# ╠═0985734e-0c1e-4e4c-8ec5-191be880a72f
# ╠═cdc62513-a1e8-4c55-a270-761b6553d806
# ╠═acc678b2-01bd-4504-ad88-f3f926cd9518
# ╠═5e7d4f3a-a7c2-4695-a211-448ed5909ad1
# ╟─6fe67c9b-fe20-42e4-b817-b31dad586e55
# ╠═a1f3534e-8c07-42b1-80ac-440bc016a652
# ╠═d04d558a-c152-43a1-8668-ab3b040e6701
# ╠═075e3c64-4dab-441c-88c4-51179121a6c9
# ╟─fc2c9720-3607-4ee2-a48c-f8e22d4404bd
# ╠═d2c83cdf-002a-47dc-81f9-22b76f183587
# ╠═08a3333b-bac8-426e-a888-9bf5269c1869
# ╟─31d32b62-4329-464e-8354-1c2875fe5801
# ╠═5c553323-3611-4614-8de3-86ea5cf8eea0
# ╠═d62b65f5-220e-45c6-a434-ac392b72ab4a
# ╠═7823bda7-5ab8-42f7-bf1c-292dbfecf178
# ╠═c3c2cfc9-e7f4-495b-bcc6-51227be2c6b5
# ╠═28c6fe28-bd42-4aba-b403-b2b0145a8e37
# ╠═1a19b769-68e2-411b-afe0-6bd2a7fb87a3
# ╠═43bc8d15-40d5-457c-84f9-57826cb4139f
# ╠═aa6cdd38-8733-425f-ac39-29d031f7c269
