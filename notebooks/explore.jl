### A Pluto.jl notebook ###
# v0.19.32

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
using Model

# ╔═╡ bbc92008-2dfb-43e3-9e16-4c60d91a2ed1
using Random

# ╔═╡ e11b7094-b8d0-458c-9e4a-8ebe3b588dfc
using Plots, PlutoUI

# ╔═╡ 9a954586-c41b-44bb-917e-2262c00b958a
using Roots

# ╔═╡ b29e58b6-dda0-4da9-b85d-d8d7c6472155
TableOfContents()

# ╔═╡ e74335e3-a230-4d11-8aad-3323961801aa
default(size = 500 .* (√2, 1), dpi = 180, titlefontsize = 12, cmap = :viridis)

# ╔═╡ 4607f0c4-027b-4402-a8b1-f98750696b6f
begin
	env = DotEnv.config()
	datapath = joinpath("..", get(env, "DATAPATH", "data/"))
	calibration = load_object(joinpath(datapath, "calibration.jld2"))
	termpath = joinpath(datapath, "terminal", "N=50_Δλ=0.08.jld2")

	V̄ = load(termpath, "V̄")
	model = load(termpath, "model")
	termpolicy = load(termpath, "policy")
end;

# ╔═╡ d51a7ae6-aaac-4ea2-8b3c-dd261222c0f8
rng = MersenneTwister(123);

# ╔═╡ c9396e59-ed2f-4f73-bf48-e94ccf6e55bd
md"""
# Terminal problem
"""

# ╔═╡ 0df664cb-2652-486f-9b83-a3bbe6314b1e
md"
## Optimal Policy

$\begin{equation}
\max_{{\color{red} \chi} \in [0, 1]} h f({\color{red} \chi}; \; y, V_i) + b^{+}({\color{red} \chi}; \; T) V_{i + \delta y} + b^{-}({\color{red} \chi}; \; T) V_{i - \delta y}
\end{equation}$
"

# ╔═╡ 45c6ee36-d1ca-464f-8e35-0b992cecb910
begin
	@unpack grid, economy = model
	
	idx = rand(rng, CartesianIndices(grid))
	Xᵢ = grid.X[idx] + [0., 0., 0.]
	Vᵢ = V̄[idx]
	Vᵢ₊ = V̄[idx + CartesianIndex(1, 0, 0)]
	Vᵢ₋ = V̄[idx - CartesianIndex(1, 0, 0)]

	b⁺(χ) = max(Model.bterminal(Xᵢ, χ, model), 0.) / grid.Δ.y
	b⁻(χ) = max(-Model.bterminal(Xᵢ, χ, model), 0.) / grid.Δ.y
	f(χ) = Model.f(χ, Xᵢ.y, Vᵢ, economy)

	function termobjective(χ)
		grid.h * f(χ) + b⁺(χ) * Vᵢ₊ + b⁻(χ) * Vᵢ₋
	end
end

# ╔═╡ 2dc66eff-a2e5-4da9-909f-f95fb5f2b6b9
md"## Terminal condition"

# ╔═╡ 15e1875d-0aa0-4f74-aa4a-6504b17d1feb
ẏ = [Model.bterminal(grid.X[idx], termpolicy[idx], model) for idx in CartesianIndices(grid)];

# ╔═╡ 4c5e5ac4-f15b-476d-b05f-e06e9b35eae4
function plotsection(F, z; zdim = 3, surf = false, 
	labels = ("\$T\$", "\$m\$", "\$y\$"), kwargs...)
	Nᵢ = size(model.grid.X)

	Ω = [range(Δ...; length = Nᵢ[i]) for (i, Δ) in enumerate(model.grid.domains)]

	xdim, ydim = filter(!=(zdim), 1:3)
    jdx = findfirst(x -> x ≥ z, Ω[zdim])

    aspect_ratio = model.grid.Δ[xdim] / model.grid.Δ[ydim]

	Z = selectdim(F, zdim, jdx)
	
    (surf ? Plots.wireframe : Plots.contourf)(
		Ω[xdim], Ω[ydim], Z'; 
        aspect_ratio, 
		xlims = model.grid.domains[xdim], 
		ylims = model.grid.domains[ydim], 
        xlabel = labels[xdim], ylabel = labels[ydim],
		clims = (minimum(Z), 0.), linewidth = 0,
		kwargs...)
end;

# ╔═╡ c5f9e376-6ab9-4a4f-960d-7dcaf8d03fb6
md"
``m``: $(@bind mterm Slider(range(model.grid.domains[2]...; length = 101), show_value = true, default = log(model.hogg.M₀)))
"

# ╔═╡ 7550e9af-e0e7-44f2-98d4-c431bdd47394
let
	vfig = plotsection(V̄, mterm; zdim = 2, title = "\$\\overline{V}\$", surf = true)
	pfig = plotsection(termpolicy, mterm; zdim = 2, title = "\$\\chi\$", surf = true)

	
	plot(vfig, pfig)
end

# ╔═╡ 6fe67c9b-fe20-42e4-b817-b31dad586e55
md"# General problem"

# ╔═╡ 24b805aa-eedc-4433-b396-d70e04a917fd
function plotpolicy(F::AbstractArray{Policy}, z; zdim = 3, surf = false, 
	labels = ("\$T\$", "\$m\$", "\$y\$"), kwargs...)
	Nᵢ = size(model.grid.X)

	Ω = [range(Δ...; length = Nᵢ[i]) for (i, Δ) in enumerate(model.grid.domains)]

	xdim, ydim = filter(!=(zdim), 1:3)
    jdx = findfirst(x -> x ≥ z, Ω[zdim])

    aspect_ratio = model.grid.Δ[xdim] / model.grid.Δ[ydim]

	Z = selectdim(F, zdim, jdx)

	routine = surf ? Plots.wireframe : Plots.contourf

	χ = first.(Z)
	α = last.(Z)
	χfig = routine(
		Ω[xdim], Ω[ydim], χ'; 
        aspect_ratio, 
		xlims = model.grid.domains[xdim], 
		ylims = model.grid.domains[ydim], 
        xlabel = labels[xdim], ylabel = labels[ydim],
		clims = zlims = (0., 1.), linewidth = 0,
		title = "\$\\chi\$",
		kwargs...)
	αfig = routine(
		Ω[xdim], Ω[ydim], α'; 
        aspect_ratio, 
		xlims = model.grid.domains[xdim], 
		ylims = model.grid.domains[ydim], 
        xlabel = labels[xdim], ylabel = labels[ydim],
		clims = zlims = (0., 1.), linewidth = 0,
		title = "\$\\alpha\$",
		kwargs...)

	return χfig, αfig
end;

# ╔═╡ 9bcfa0d7-0442-40f0-b63f-7d39f38c1310
begin
	simpath = joinpath(datapath, "total", "N=50_Δλ=0.08.jld2")
	file = jldopen(simpath, "r")
	timesteps = keys(file)
	V = Array{Float64}(undef, 50, 50, 50, length(timesteps))
	policy = Array{Policy}(undef, 50, 50, 50, length(timesteps))

	for t in timesteps
		V[:, :, :, tryparse(Int, t) + 1] = file[t]["V"]
		policy[:, :, :, tryparse(Int, t) + 1] = file[t]["policy"]
	end
	
	close(file)
end

# ╔═╡ 754dea44-74f4-471d-87b4-d991134b378f
md"
``m``: $(@bind m Slider(range(model.grid.domains[2]...; length = 101), show_value = true, default = log(model.hogg.M₀)))

``t``: $(@bind t Slider(0:(size(V, 4) - 1), show_value = true, default = size(V, 4) - 0))
"

# ╔═╡ 7fbce055-705c-4d58-8c1a-5fffbbe66037
let
	plotsection(V[:, :, :, t + 1], m; zdim = 2, title = "\$\\overline{V}\$", surf = true)
end

# ╔═╡ 285a3e22-e1cd-4c94-939f-6d011c8db8ca
let
	χfig, αfig = plotpolicy(policy[:, :, :, t + 1], m; zdim = 2, surf = true)

	plot(χfig, αfig)
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
# ╠═4607f0c4-027b-4402-a8b1-f98750696b6f
# ╠═d51a7ae6-aaac-4ea2-8b3c-dd261222c0f8
# ╟─c9396e59-ed2f-4f73-bf48-e94ccf6e55bd
# ╟─0df664cb-2652-486f-9b83-a3bbe6314b1e
# ╟─45c6ee36-d1ca-464f-8e35-0b992cecb910
# ╟─2dc66eff-a2e5-4da9-909f-f95fb5f2b6b9
# ╠═15e1875d-0aa0-4f74-aa4a-6504b17d1feb
# ╠═4c5e5ac4-f15b-476d-b05f-e06e9b35eae4
# ╟─c5f9e376-6ab9-4a4f-960d-7dcaf8d03fb6
# ╟─7550e9af-e0e7-44f2-98d4-c431bdd47394
# ╟─6fe67c9b-fe20-42e4-b817-b31dad586e55
# ╠═24b805aa-eedc-4433-b396-d70e04a917fd
# ╠═9bcfa0d7-0442-40f0-b63f-7d39f38c1310
# ╟─754dea44-74f4-471d-87b4-d991134b378f
# ╠═7fbce055-705c-4d58-8c1a-5fffbbe66037
# ╠═285a3e22-e1cd-4c94-939f-6d011c8db8ca
