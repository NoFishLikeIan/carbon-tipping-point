### A Pluto.jl notebook ###
# v0.19.46

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
using JLD2, FastClosures, Roots

# ╔═╡ 4d074922-3d70-442b-be61-18f99f213d3d
using DataStructures: dequeue!

# ╔═╡ ee4b57ae-517b-4b3a-9ab0-e328cc97a121
using DotEnv

# ╔═╡ c7e2f611-07e4-4c39-b101-d20b3f71fedb
using Distributed

# ╔═╡ bfc6af5e-a261-4d7e-9a16-f4ab54c6e1ca
using PlutoUI

# ╔═╡ 9655de0d-73f7-4332-85d9-974a73b4fce1
using Model, Grid

# ╔═╡ 62a3b757-cf0b-4f8d-bdb8-7edc7ba04d47
using Optim

# ╔═╡ ddd1a388-76fe-482f-ad8e-fbc1096f2d43
begin
	using Plots
	default(size = 500 .* (√2, 1), dpi = 180, linewidth = 2, cmap = :viridis)

	using LaTeXStrings, Printf
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
	Plotting = ingredients("../scripts/plotting/utils.jl")
	Chain = ingredients("../scripts/markov/chain.jl")
	Terminal = ingredients("../scripts/markov/terminal.jl")
	Backward = ingredients("../scripts/markov/backward.jl")
end;

# ╔═╡ e11493be-9406-4fe8-9c85-ac8deb2d1953
begin
	reltoroot = path -> joinpath("..", path)
	env = DotEnv.config(reltoroot(".env"))
	
	DATAPATH = get(env, "DATAPATH", "") |> reltoroot
	SIMULATIONPATH = get(env, "SIMULATIONPATH", "")
	datapath = joinpath(DATAPATH, SIMULATIONPATH)
	
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	hogg = Hogg()
	economy = Economy()
	preferences = EpsteinZin()

	damages = GrowthDamages()
	albedo = Albedo(1.5)

	model = TippingModel(albedo, hogg, preferences, damages, economy, calibration);
end;

# ╔═╡ b38949ca-4432-4b8f-be02-3d96e5b1fce0
begin
	N = 51
	Tdev = (0., 7.)
	Tdomain = hogg.Tᵖ .+ Tdev;
	mdomain = mstable.(Tdomain, hogg)
	G = RegularGrid([Tdomain, mdomain], N)

	Tspace = range(Tdomain...; length = N)
	mspace = range(mdomain...; length = N)
	unit = range(1e-3, 1 - 1e-3; step = 1e-3)

	F̄, terminalpolicy = Saving.loadterminal(model; datapath)
	Gterm = terminalgrid(N, model);

	policy = Array{Float64}(undef, size(G)..., 2)
	policy[:, :, 1] .= interpolateovergrid(Gterm, G, terminalpolicy)
	policy[:, :, 2] .= γ(economy.τ, calibration)
	
	F = interpolateovergrid(Gterm, G, F̄)
end;

# ╔═╡ 28f9e438-f800-4fee-a216-6daa4ead8da6
begin # Plotting utilities
	getbyspace(T, m) = getbyspace(Point(T, m))
	function getbyspace(x::Point)
		i = findfirst(≥(x.T), Tspace)
		j = findfirst(≥(x.m), mspace)
		return CartesianIndex(i, j)
	end

	mticks = range(mdomain...; length=6)
	Mticks = (mticks, round.(exp.(mticks), digits = 2))

	Tticks = Plotting.makedeviationtickz(Tdev..., model; step = 1)

	satres = Optim.optimize(m -> -δₘ(exp(m[1]), hogg), [log(hogg.:M₀)]);

	αmax = δₘ(exp(first(Optim.minimizer(satres))), hogg) + γ(economy.τ, calibration)
end;

# ╔═╡ 4f53f52a-f3b7-48fb-b63c-9fbc7f9dd65e
begin
	Ffig = contourf(mspace, Tspace, log.(F); xticks = Mticks, yticks = Tticks, ylabel = L"T_t - T^p", xlabel = L"M_t", title = L"$F_{\tau}(T, M)$", xlims = mdomain, ylims = Tdomain, cbar = false)

	Tdense = range(Tdomain...; length = 101)
	nullcline = mstable.(Tdense, hogg, Albedo(1.5))

	plot!(Ffig, nullcline, Tdense; c = :white, label = false)
end

# ╔═╡ 015922f1-c954-42db-887e-4499e5dbca59
md"# Backward costs"

# ╔═╡ 5aa075ef-8ad2-4d52-b4da-d99d687a7d4e
begin
	function makeobjective(T, m; t = economy.τ)
		idx = getbyspace(T, m)
		makeobjective(idx; t)
	end
	function makeobjective(idx; t = economy.τ)
		Xᵢ = G.X[idx]
	
		objective = @closure u -> begin
			F′, Δt = Chain.markovstep(t, idx, F, u, model, G)
			Chain.cost(F′, t, Xᵢ, Δt, u, model)
		end
	
		return objective
	end
end;

# ╔═╡ 0169acc9-2d30-4e80-a38b-a0cbc5af15dc
md"
- ``T =`` $(@bind Tfig Slider(Tspace, default = hogg.T₀, show_value = true))
- ``m =`` $(@bind mfig Slider(mspace, default = log(hogg.M₀), show_value = true))
- ``t =`` $(@bind tfig Slider(1:economy.τ, default = economy.τ, show_value = true))
"

# ╔═╡ ec33fe87-f4fd-4454-bfa7-e644bb89f344
begin
	ᾱ = γ(tfig, calibration) + δₘ(exp(mfig), hogg)
	idx = getbyspace(Tfig, mfig)
	obj = makeobjective(Tfig, mfig; t = tfig)

	χ = terminalpolicy[first(idx.I)]
	u₀ = [χ, 1e-3]
	od = TwiceDifferentiable(obj, u₀; autodiff = :forward)
	cons = TwiceDifferentiableConstraints([0., 0.], [1., ᾱ])

	resminimisation = Optim.optimize(od, cons, u₀, IPNewton())
	u = Optim.minimizer(resminimisation)

	!Optim.converged(resminimisation) && @warn "Not converged"
end;

# ╔═╡ fb36873a-3db7-439e-955f-24e0725bd6b3
begin
	cspace = range(0.4, 0.7; length = 51)
	aspace = range(0., αmax; length = 51)

	Fobjfig = deepcopy(Ffig)
	scatter!(Fobjfig, [mfig], [Tfig], c = :white, label = false, markersize = 5)

	objfig = contourf(cspace, aspace, (χ, α) -> log(obj([χ, α])); 
		ylims = (0, αmax), xlims = extrema(cspace),
		xlabel = L"\chi", ylabel = L"\alpha", c = :Reds, linewidth = 1, cbar = false
	)

	hline!(objfig, [ᾱ]; linestyle = :dash, label = false, color = :white)
	scatter!(objfig, u[[1]], u[[2]]; label = false, c = :green)
	
	plot(objfig, Fobjfig; size = 410 .* (2√2, 1), margins = 5Plots.mm)
end

# ╔═╡ b8b54945-ab69-4bfc-862f-9498aa0c30fc
function updateᾱ!(constraints::TwiceDifferentiableConstraints, ᾱ)
    constraints.bounds.bx[4] = ᾱ
end;

# ╔═╡ 78a46a5e-7420-4e74-acb1-693ba5664f3b
begin
	abatement = similar(F)
	constraints = TwiceDifferentiableConstraints([0., 0.], [1., 1.])
	
	for idx in CartesianIndices(G)
		χ = terminalpolicy[first(idx.I)]
		u₀ = [χ, 1e-3]

		obj = makeobjective(idx)
		od = TwiceDifferentiable(obj, u₀; autodiff = :forward)
		updateᾱ!(constraints, γ(economy.τ, calibration) + δₘ(exp(G.X[idx].m), hogg))

		resminimisation = Optim.optimize(od, cons, u₀, IPNewton())
		u = Optim.minimizer(resminimisation)

		abatement[idx] = u[2] 
	end
end;

# ╔═╡ 1d27e09e-e3dd-4793-9059-1587835a3885
begin
	abatfig = heatmap(mspace, Tspace, abatement; xticks = Mticks, yticks = Tticks, ylabel = L"T_t - T^p", xlabel = L"M_t", title = L"$F_{\tau}(T, M)$", xlims = mdomain, ylims = Tdomain)

	plot!(abatfig, nullcline, Tdense; c = :white, label = false)
end

# ╔═╡ Cell order:
# ╟─e29f796c-c57c-40c3-988a-b7d9295c3dac
# ╟─c4befece-4e47-11ef-36e3-a97f8e06d12b
# ╟─fd94250f-950c-4d37-94ad-ca7d73637d08
# ╠═2b3c9a7f-8078-415d-b132-999a01aca419
# ╠═4d074922-3d70-442b-be61-18f99f213d3d
# ╠═ee4b57ae-517b-4b3a-9ab0-e328cc97a121
# ╠═c7e2f611-07e4-4c39-b101-d20b3f71fedb
# ╠═bfc6af5e-a261-4d7e-9a16-f4ab54c6e1ca
# ╠═9655de0d-73f7-4332-85d9-974a73b4fce1
# ╠═62a3b757-cf0b-4f8d-bdb8-7edc7ba04d47
# ╠═ddd1a388-76fe-482f-ad8e-fbc1096f2d43
# ╠═e0877f7e-6247-4c22-a1c7-fdc42ad4dec2
# ╠═e11493be-9406-4fe8-9c85-ac8deb2d1953
# ╠═b38949ca-4432-4b8f-be02-3d96e5b1fce0
# ╠═28f9e438-f800-4fee-a216-6daa4ead8da6
# ╟─4f53f52a-f3b7-48fb-b63c-9fbc7f9dd65e
# ╟─015922f1-c954-42db-887e-4499e5dbca59
# ╠═5aa075ef-8ad2-4d52-b4da-d99d687a7d4e
# ╟─0169acc9-2d30-4e80-a38b-a0cbc5af15dc
# ╟─ec33fe87-f4fd-4454-bfa7-e644bb89f344
# ╠═fb36873a-3db7-439e-955f-24e0725bd6b3
# ╠═b8b54945-ab69-4bfc-862f-9498aa0c30fc
# ╠═78a46a5e-7420-4e74-acb1-693ba5664f3b
# ╠═1d27e09e-e3dd-4793-9059-1587835a3885
