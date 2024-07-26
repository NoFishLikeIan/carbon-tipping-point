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

# ╔═╡ 6d43d072-4b34-11ef-3e45-8fde68196131
begin # Use local module
	import Pkg
  	Pkg.activate(Base.current_project())
    Pkg.instantiate()
end

# ╔═╡ b447bdc5-ea85-462c-a1f0-c8c8d5a3f1e9
using PlutoUI; TableOfContents()

# ╔═╡ 5eb83b47-5f48-463d-8abb-21eaf36dbc25
using JLD2, FastClosures

# ╔═╡ cbd38547-eea1-4214-88c7-944c1aca82c2
using Model, Grid

# ╔═╡ f6bab433-5d9d-4516-9c24-59c5441c06eb
begin
	using Plots

	default(size = 500 .* (√2, 1), dpi = 180, linewidth = 1.5)
end

# ╔═╡ bfa6364c-7376-449e-90bc-11a37334092a
using DifferentialEquations, DifferentialEquations.EnsembleAnalysis

# ╔═╡ 35f99656-f68a-455c-9042-b5afc5e7a4a8
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

# ╔═╡ 44eee19a-595a-42fe-9302-0f19df42388a
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
end;

# ╔═╡ 729c4427-986c-48b2-ae05-3b7b51c7d6bd
md"# Setup"

# ╔═╡ 576db675-6ca4-4f1e-8782-5bc39ef335e6
DATAPATH = "../data";

# ╔═╡ 542e01e0-bbce-462b-9c43-6d375adb6bd5
Saving = ingredients("../scripts/utils/saving.jl");

# ╔═╡ 2ba00290-bc0b-4f73-91fe-8fcbde9b1589
Simulating = ingredients("../scripts/utils/simulating.jl");

# ╔═╡ abc6d59d-6b20-4193-ab50-53facae969e1
begin # Default parameters
	preferences = EpsteinZin();
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"));
	economy = Economy()
	hogg = Hogg()
end;

# ╔═╡ cfad70a3-73d0-4d2f-9180-3f2c6322b17d
md"# Analysis"

# ╔═╡ 623654ff-251c-4570-a870-009933e62197
begin # Parameters
	damage = LevelDamages()
	albedo = Albedo(λ₂ = Albedo().λ₁)
	jump = Jump()

	model = TippingModel(albedo, preferences, damage, economy, hogg, calibration)
	# model = JumpModel(jump, preferences, damage, economy, hogg, calibration)

	N = 51
end;

# ╔═╡ 891cac99-6427-47f2-8956-d4eb7817ea54
begin
	G = constructdefaultgrid(N, model)
	result = Saving.loadtotal(model, G; datapath = DATAPATH)
	Fitp, χitp, αitp = Simulating.buildinterpolations(result, G)
end;

# ╔═╡ 7787a015-b714-4107-829e-ac2e7224a864
begin
	Tspace = range(G.domains[1]...; length = size(G, 1))
	mspace = range(G.domains[2]...; length = size(G, 2))
end;

# ╔═╡ 78c11f3a-1325-46e5-974b-ae67b9637834
md"
## Explore value function

`t =` $(@bind tfig Slider(range(extrema(first(result))...; step = 1/4), default = 0., show_value = true))
"

# ╔═╡ 39f4e853-debb-4224-bb30-7a85bd8a35fb
let
	fₜ = @closure (T, m) -> log(Fitp(T, m, tfig))
	zlims = log.(extrema(Fitp.coefs))
	
	contourf(Tspace, mspace, fₜ; xlabel = "\$T\$", ylabel = "\$m\$", title = "\$\\log(F_t)\$ at \$t = $tfig\$", clims = zlims, linewidth = 0.)
end

# ╔═╡ 033b0310-2df1-4613-81f8-39274d2318ba
let
	χₜ = @closure (T, m) -> χitp(T, m, tfig)
	αₜ = @closure (T, m) -> αitp(T, m, tfig)
	
	consfig = contourf(Tspace, mspace, χₜ; xlabel = "\$T\$", ylabel = "\$m\$", title = "Time \$t = $tfig\$; \$\\chi\$", clims = (0, 1), linewidth = 0.)

	abatfig = contourf(Tspace, mspace, αₜ; xlabel = "\$T\$", ylabel = "\$m\$", title = "\$\\alpha\$", linewidth = 0.)

	plot(consfig, abatfig; size = 500 .* (2√2, 1), margins = 10Plots.mm)
end

# ╔═╡ 0df8cf8d-4134-47ec-b3d2-c98248ba859f
md"## Simulate"

# ╔═╡ a40b7065-f7ca-47ae-a05c-982de4a3c3b4


# ╔═╡ Cell order:
# ╟─35f99656-f68a-455c-9042-b5afc5e7a4a8
# ╟─44eee19a-595a-42fe-9302-0f19df42388a
# ╟─6d43d072-4b34-11ef-3e45-8fde68196131
# ╠═b447bdc5-ea85-462c-a1f0-c8c8d5a3f1e9
# ╟─729c4427-986c-48b2-ae05-3b7b51c7d6bd
# ╠═576db675-6ca4-4f1e-8782-5bc39ef335e6
# ╠═5eb83b47-5f48-463d-8abb-21eaf36dbc25
# ╠═cbd38547-eea1-4214-88c7-944c1aca82c2
# ╠═f6bab433-5d9d-4516-9c24-59c5441c06eb
# ╠═bfa6364c-7376-449e-90bc-11a37334092a
# ╠═542e01e0-bbce-462b-9c43-6d375adb6bd5
# ╠═2ba00290-bc0b-4f73-91fe-8fcbde9b1589
# ╠═abc6d59d-6b20-4193-ab50-53facae969e1
# ╟─cfad70a3-73d0-4d2f-9180-3f2c6322b17d
# ╠═623654ff-251c-4570-a870-009933e62197
# ╠═891cac99-6427-47f2-8956-d4eb7817ea54
# ╠═7787a015-b714-4107-829e-ac2e7224a864
# ╟─78c11f3a-1325-46e5-974b-ae67b9637834
# ╟─39f4e853-debb-4224-bb30-7a85bd8a35fb
# ╟─033b0310-2df1-4613-81f8-39274d2318ba
# ╟─0df8cf8d-4134-47ec-b3d2-c98248ba859f
# ╠═a40b7065-f7ca-47ae-a05c-982de4a3c3b4
