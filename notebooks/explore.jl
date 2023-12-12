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

# ╔═╡ d8a70e6a-e977-4335-8698-e4aee9220a01
using Optim

# ╔═╡ b29e58b6-dda0-4da9-b85d-d8d7c6472155
TableOfContents()

# ╔═╡ e74335e3-a230-4d11-8aad-3323961801aa
default(size = 500 .* (√2, 1), dpi = 180, titlefontsize = 12, cmap = :viridis)

# ╔═╡ 4607f0c4-027b-4402-a8b1-f98750696b6f
begin
	env = DotEnv.config()
	datapath = joinpath("..", get(env, "DATAPATH", "data/"))
	@load joinpath(datapath, "calibration.jld2") calibration
	@load joinpath(datapath, "terminal.jld2") V̄ policy model
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

# ╔═╡ 7e333750-df9b-47e6-b0d4-1ca8dd26339d
plot(0:0.01:1, termobjective)

# ╔═╡ 69b1534c-2dd3-47cf-9640-f4f044e059b3
contourf(
	0:0.01:1, range(extrema(V̄)...; length = 101),
	(χ, v) -> Model.f(χ, Xᵢ.y, v, Economy(ρ = 0.02));
	linewidth = 0)

# ╔═╡ 2dc66eff-a2e5-4da9-909f-f95fb5f2b6b9
md"## Terminal condition"

# ╔═╡ 15e1875d-0aa0-4f74-aa4a-6504b17d1feb
ẏ = [Model.bterminal(grid.X[idx], policy[idx], model) for idx in CartesianIndices(grid)];

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
``m``: $(@bind m Slider(range(model.grid.domains[2]...; length = 101), show_value = true, default = log(model.hogg.M₀)))
"

# ╔═╡ 7550e9af-e0e7-44f2-98d4-c431bdd47394
let
	vfig = plotsection(V̄, m; zdim = 2, title = "\$\\overline{V}\$", surf = true)
	pfig = plotsection(policy, m; zdim = 2, title = "\$\\chi\$", surf = true)

	
	plot(vfig, pfig)
end

# ╔═╡ 6fe67c9b-fe20-42e4-b817-b31dad586e55
md"""
# General problem
## Terminal policy
"""

# ╔═╡ 60aa84fd-8f83-4dd6-a86d-0b0d0630f158
begin
	L, R = extrema(CartesianIndices(grid))
    VᵢT₊, VᵢT₋ = V̄[min(idx + Model.I[1], R)], V̄[max(idx - Model.I[1], L)]
    Vᵢm₊ = V̄[min(idx + Model.I[2], R)]
    Vᵢy₊, Vᵢy₋ = V̄[min(idx + Model.I[3], R)], V̄[max(idx - Model.I[3], L)]

	t = 120
	γₜ = Model.γ(t, economy, calibration)
end;

# ╔═╡ 21726b66-3f36-4d53-9722-58177385759c
function objective!(z, ∇, H, u)
        χ, α = u
        bᵢ = Model.b(t, Xᵢ, χ, α, model) / model.grid.Δ.y
        bsgn = sign(bᵢ)
        Vᵢy = ifelse(bᵢ > 0, Vᵢy₊, Vᵢy₋)

        M = exp(Xᵢ.m)
        Aₜ = Model.A(t, economy)
        εₜ = Model.ε(t, M, α, model)
        εₜ′ = Model.ε′(t, M, model)

        fᵢ, Y∂fᵢ, Y²∂²fᵢ = Model.epsteinzinsystem(χ, Xᵢ.y, Vᵢ, economy) .* grid.h

        if !isnothing(∇) 
            ∇[1] = -Y∂fᵢ - bsgn * Model.ϕ′(t, χ, economy) * Vᵢy
            ∇[2] = Vᵢm₊ + bsgn * Aₜ * Model.β′(t, εₜ, model.economy) * εₜ′ * Vᵢy
        end

        if !isnothing(H)
            H[1, 2] = 0.
            H[2, 1] = 0.
            
            H[1, 1] = -Y²∂²fᵢ + bsgn * economy.κ * Aₜ^2 * Vᵢy
            H[2, 2] = bsgn * Aₜ * (εₜ′)^2 * exp(-economy.ωᵣ * t) * Vᵢy
        end
        
        if !isnothing(z)
            z = α * Vᵢm₊ - abs(bᵢ) * Vᵢy - fᵢ
            return z
        end
    end

# ╔═╡ b6225e9c-2b82-4768-ab02-c1c9a09bcb11
begin
	p₀ = [1e-3, 1e-3]
	dc = TwiceDifferentiableConstraints([0., 0.], [1., γₜ])
	df = TwiceDifferentiable(Optim.only_fgh!(objective!), p₀)
	res = optimize(df, dc, p₀, IPNewton())
end

# ╔═╡ 61fba0ff-4e94-472d-a69c-9799e7c4698f
let
	Cspace = range(0, 1; length = 31)
	Aspace = range(0, γₜ; length = 31)

	contourf(Cspace, Aspace, (χ, α) -> Model.b(t, Xᵢ, χ, α, model); 
		ylabel = "\$\\alpha\$", xlabel = "\$\\chi\$",
		camera = (20, 65), cbar = false, 
		ylims = extrema(Aspace), xlims = extrema(Cspace),
		linewidth = 0
	)

	χᵢ, αᵢ = Optim.minimizer(res)

	scatter!([χᵢ], [αᵢ]; c = :red, label = false)
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
# ╟─7e333750-df9b-47e6-b0d4-1ca8dd26339d
# ╠═69b1534c-2dd3-47cf-9640-f4f044e059b3
# ╟─2dc66eff-a2e5-4da9-909f-f95fb5f2b6b9
# ╠═15e1875d-0aa0-4f74-aa4a-6504b17d1feb
# ╠═4c5e5ac4-f15b-476d-b05f-e06e9b35eae4
# ╟─c5f9e376-6ab9-4a4f-960d-7dcaf8d03fb6
# ╟─7550e9af-e0e7-44f2-98d4-c431bdd47394
# ╟─6fe67c9b-fe20-42e4-b817-b31dad586e55
# ╠═d8a70e6a-e977-4335-8698-e4aee9220a01
# ╠═60aa84fd-8f83-4dd6-a86d-0b0d0630f158
# ╠═21726b66-3f36-4d53-9722-58177385759c
# ╠═b6225e9c-2b82-4768-ab02-c1c9a09bcb11
# ╠═61fba0ff-4e94-472d-a69c-9799e7c4698f
