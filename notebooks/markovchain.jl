### A Pluto.jl notebook ###
# v0.19.45

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

# ╔═╡ c4befece-4e47-11ef-36e3-a97f8e06d12b
begin # Use local module
	import Pkg
  	Pkg.activate(Base.current_project())
    Pkg.instantiate()
end

# ╔═╡ 2b3c9a7f-8078-415d-b132-999a01aca419
using JLD2, FastClosures

# ╔═╡ bfc6af5e-a261-4d7e-9a16-f4ab54c6e1ca
using PlutoUI

# ╔═╡ 9655de0d-73f7-4332-85d9-974a73b4fce1
using Model, Grid

# ╔═╡ ddd1a388-76fe-482f-ad8e-fbc1096f2d43
begin
	using Plots
	default(size = 500 .* (√2, 1), dpi = 180, linewidth = 2, cmap = :viridis)
end

# ╔═╡ e29f796c-c57c-40c3-988a-b7d9295c3dac
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

# ╔═╡ fd94250f-950c-4d37-94ad-ca7d73637d08
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

# ╔═╡ e0877f7e-6247-4c22-a1c7-fdc42ad4dec2
begin
	Saving = ingredients("../scripts/utils/saving.jl")
	Terminal = ingredients("../scripts/terminal.jl")
end;

# ╔═╡ e11493be-9406-4fe8-9c85-ac8deb2d1953
begin
	DATAPATH = "../data"
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	hogg = Hogg()
	economy = Economy(τ = 500.)
	preferences = EpsteinZin()
end;

# ╔═╡ 31baffdb-ad83-49a9-a01c-4329626783b2
begin
	Δλ = 0.08
	damages = GrowthDamages()
	albedo = Albedo(λ₂ = Albedo().λ₁ - Δλ)

	model = TippingModel(albedo, preferences, damages, economy, hogg, calibration)
end;

# ╔═╡ b38949ca-4432-4b8f-be02-3d96e5b1fce0
begin
	N = 51
	G = constructdefaultgrid(N, model)

	Tspace = range(G.domains[1]...; length = size(G, 1))
	mspace = range(G.domains[2]...; length = size(G, 2))
	unit = range(1e-3, 1 - 1e-3; step = 1e-3)

	F₀ = ones(size(G)); pol₀ = ones(size(G)) ./ 2;
end;

# ╔═╡ 28f9e438-f800-4fee-a216-6daa4ead8da6
fromspacetoidx = @closure (T, m) -> begin
	i = findfirst(≥(T), Tspace)
	j = findfirst(≥(m), mspace)

	return CartesianIndex(i, j)
end;

# ╔═╡ fe1331d2-5ed2-4507-991c-10a23fccbb50
md"# Markov Chain of Terminal Problem"

# ╔═╡ b852741b-4857-40f1-82b6-52504827e819
begin
	F̄, terminalpolicy = Saving.loadterminal(model, G; datapath=DATAPATH)
	# F̄ = copy(F₀)
end;

# ╔═╡ cdb1d34f-ff03-49b1-b1d1-75db7aad46c7
terminalmarkovstep = @closure (T, m) -> begin
	idx = fromspacetoidx(T, m)
	Terminal.terminalmarkovstep(idx, F̄, model, G)
end;

# ╔═╡ ac633771-3b7f-4cb4-8f41-210f676883c4
begin
	terminalcosts(T, m, χ) = terminalcosts(fromspacetoidx(T, m), χ)
	function terminalcosts(idx, χ)
		Fᵢ′, Δt = Terminal.terminalmarkovstep(idx, F̄, model, G)
		Tᵢ = G.X[idx].T
		Terminal.terminalcost(Fᵢ′, Tᵢ, Δt, χ, model)
	end
end;

# ╔═╡ dcf00953-09b5-4cfd-95ce-8a6d882d9174
let
	timefig = heatmap(mspace, Tspace, (m, T) -> terminalmarkovstep(T, m) |> last; xlabel = "\$m\$", ylabel = "\$T\$", title = "\$\\Delta t\$", clims = (0, Inf), cmap = :Reds)
	markovfig = heatmap(mspace, Tspace, (m, T) -> terminalmarkovstep(T, m) |> first |> log; xlabel = "\$m\$", ylabel = "\$T\$", cmap = :coolwarm, title = "\$\\log \\mathbb{E}[F_{t + \\Delta t}]\$")

	plot(timefig, markovfig; size = 400 .* (2√2, 1), margins = 10Plots.mm)
end

# ╔═╡ 837f8bac-8f45-443b-8b59-10deb045c4e1
md"``\chi =`` $(@bind χ Slider(unit, default = 0.5, show_value = true))"

# ╔═╡ 072d27e3-2d6f-43a4-b319-e345631b57ad
let
	heatmap(mspace, Tspace, (m, T) -> terminalcosts(T, m, χ) |> log; xlabel = "\$m\$", ylabel = "\$T\$", cmap = :coolwarm, title = "\$\\log \\bar{F} \\mid \\chi = $χ\$", clims = (-6, 6))
end

# ╔═╡ 30c3dc7a-9b6d-4f94-881a-f29607edbdff
begin
	iterations = 100
	Fpath = Array{Float64}(undef, size(G, 1), size(G, 2), iterations + 1)
	χpath = similar(Fpath)

	Fₖ = copy(F₀); χₖ = copy(pol₀)
	Fpath[:, :, 1] .= F₀
	χpath[:, :, 1] .= pol₀
	
	for k in 1:iterations
		Terminal.terminaljacobi!(Fₖ, χₖ, model, G)
		Fpath[:, :, k + 1] = Fₖ
		χpath[:, :, k + 1] = χₖ
	end
end

# ╔═╡ 0fd33a23-b2ec-4857-9a13-4e06d43b9d24
md"Iteration ``k =`` $(@bind k Slider(0:iterations, default = 0, show_value = true))"

# ╔═╡ d21ccdc8-f43d-45bc-9409-8008b2b2db17
let
	Fₖ = @view Fpath[:, :, k + 1]
	χₖ = @view χpath[:, :, k + 1]
	
	Ffig = heatmap(mspace, Tspace, (m, T) -> Fₖ[fromspacetoidx(T, m)] |> log; xlabel = "\$m\$", ylabel = "\$T\$", cmap = :coolwarm, title = "\$\\log F_{$k}\$");
	χfig = heatmap(mspace, Tspace, (m, T) -> χₖ[fromspacetoidx(T, m)]; xlabel = "\$m\$", ylabel = "\$T\$", cmap = :Greens, title = "\$χ_{$k}\$");

	plot(Ffig, χfig; size = 400 .* (2√2, 1), margins = 10Plots.mm)
end

# ╔═╡ ebff6eea-2014-470f-a3ee-fffbb7603cfd
let
	plot(Tspace, T -> χₖ[fromspacetoidx(T, mstable(T, hogg, albedo))]; xlabel = "\$T\$", label = "\$\\chi(T, \\bar{m}(T)) \$", c = :black)
	plot!(Tspace[5:(N - 5)], T -> χₖ[fromspacetoidx(T, min(0.5 + mstable(T, hogg, albedo), maximum(mspace)))]; xlabel = "\$T\$", label = "\$\\chi(T, \\bar{m}(T) + 0.5) \$", c = :darkred)
	plot!(Tspace[5:(N - 5)], T -> χₖ[fromspacetoidx(T, max(-0.5 + mstable(T, hogg, albedo), minimum(mspace)))]; xlabel = "\$T\$", label = "\$\\chi(T, \\bar{m}(T) - 0.5) \$", c = :darkgreen)
end

# ╔═╡ Cell order:
# ╟─e29f796c-c57c-40c3-988a-b7d9295c3dac
# ╟─c4befece-4e47-11ef-36e3-a97f8e06d12b
# ╟─fd94250f-950c-4d37-94ad-ca7d73637d08
# ╠═2b3c9a7f-8078-415d-b132-999a01aca419
# ╠═bfc6af5e-a261-4d7e-9a16-f4ab54c6e1ca
# ╠═9655de0d-73f7-4332-85d9-974a73b4fce1
# ╠═ddd1a388-76fe-482f-ad8e-fbc1096f2d43
# ╠═e0877f7e-6247-4c22-a1c7-fdc42ad4dec2
# ╠═e11493be-9406-4fe8-9c85-ac8deb2d1953
# ╠═31baffdb-ad83-49a9-a01c-4329626783b2
# ╠═b38949ca-4432-4b8f-be02-3d96e5b1fce0
# ╠═28f9e438-f800-4fee-a216-6daa4ead8da6
# ╟─fe1331d2-5ed2-4507-991c-10a23fccbb50
# ╠═b852741b-4857-40f1-82b6-52504827e819
# ╠═cdb1d34f-ff03-49b1-b1d1-75db7aad46c7
# ╠═ac633771-3b7f-4cb4-8f41-210f676883c4
# ╟─dcf00953-09b5-4cfd-95ce-8a6d882d9174
# ╟─837f8bac-8f45-443b-8b59-10deb045c4e1
# ╠═072d27e3-2d6f-43a4-b319-e345631b57ad
# ╠═30c3dc7a-9b6d-4f94-881a-f29607edbdff
# ╟─0fd33a23-b2ec-4857-9a13-4e06d43b9d24
# ╠═d21ccdc8-f43d-45bc-9409-8008b2b2db17
# ╠═ebff6eea-2014-470f-a3ee-fffbb7603cfd
