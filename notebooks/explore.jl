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

	using UnPack, JLD2, DotEnv
end

# ╔═╡ f3d4f91d-ebac-43cb-9789-df38f9a87a8c
using Model, Utils

# ╔═╡ e11b7094-b8d0-458c-9e4a-8ebe3b588dfc
using Plots, PlutoUI

# ╔═╡ 7d0297b2-a0e7-45b6-b573-cb6015b1c283
using Optim

# ╔═╡ e74335e3-a230-4d11-8aad-3323961801aa
default(size = 500 .* (√2, 1), dpi = 180, titlefontsize = 12)

# ╔═╡ 713208a4-2d81-4db5-afd4-9b99816518dc
defcmap = cgrad(:YlGnBu, rev = true);

# ╔═╡ b3ac7b02-2aa5-4da9-9459-0ea24d703697
PLOTPATH = "../plots/presentation/" 

# ╔═╡ 3c65c27d-ee8f-45cf-81ec-5183e1949a59
md"## Epstein-Zin"

# ╔═╡ 2dc66eff-a2e5-4da9-909f-f95fb5f2b6b9
md"## Import terminal condition"

# ╔═╡ 4607f0c4-027b-4402-a8b1-f98750696b6f
datapath = joinpath("..", get(DotEnv.config(), "DATAPATH", "data/")) 

# ╔═╡ 5d878f4a-279e-424f-8da5-d9de5e491a8d
@load joinpath(datapath, "calibration.jld2") calibration;

# ╔═╡ 002cf9d6-369a-48e2-8797-31f251637866
@load joinpath(datapath, "terminal.jld2") V̄;

# ╔═╡ ddea2917-4271-4c72-b9f0-4e1aa6be42de
economy = Model.Economy(); hogg = Model.Hogg(); albedo = Model.Albedo();

# ╔═╡ 7ca9d16e-a543-429e-b126-29db7bb51796
begin
	consumption = range(0, economy.Ȳ; length = 101)
	utility = range(-2, 0; length = 101)
	
	contourf(consumption, utility, (c, u) -> Model.f(c, 0, u, economy), c = :viridis, linewidth = 0, aspect_ratio = economy.Ȳ / 2, xlims = extrema(consumption), ylims = extrema(utility), xlabel = "\$c\$", ylabel = "\$u\$", title = "\$f(c, u)\$")
end

# ╔═╡ a948f29f-4ad7-4f59-a624-6dae9c5c2f42
instance = (economy, hogg, albedo);

# ╔═╡ a85e9db2-88eb-409b-b44f-9e7c8820e1b9
begin
	n₁, n₂ = size(V̄)
	domain = [
		(hogg.T₀, hogg.T̄, n₁), 
		(log(hogg.M₀), log(hogg.M₀ + 400f0), n₂), 
		(log(economy.Y̲), log(economy.Ȳ), n₂)
	]
	Ω = Utils.makegrid(domain)
end;

# ╔═╡ 5ba540ee-6ac3-4265-82bd-1a9a840bc927
Model.Mstable(287.15f0 + 10f0, hogg, albedo)

# ╔═╡ dc2cd7ba-63ef-44aa-b9f4-e6bd003b8cf3
hogg.T̄ - hogg.T₀

# ╔═╡ f08b07ce-c488-4ff7-adb9-18752a0a3a2e
ar = diff(collect(extrema(Ω[1]))) / diff(collect(extrema(Ω[3])));

# ╔═╡ 51512c63-296d-4f4a-aa97-97e91df1304f
begin
	termfig = contourf(Ω[1], Ω[3], V̄'; 
		xlabel = "\$T\$", ylabel = "\$y\$", title = "\$\\bar{V}(T, y)\$",
		linewidth = 0, aspect_ratio = ar, 
		cmap = defcmap, 
		xlims = extrema(Ω[1]), ylims = extrema(Ω[3]),
		cbar = false, dpi = 180
	)
end

# ╔═╡ 51700462-1e9f-4548-aa07-d82f0fcd20cb
savefig(termfig, joinpath(PLOTPATH, "termfig.png"))

# ╔═╡ f49ffc11-6204-4bdc-a17f-80c4f7e8cdce
md"
## Optimisation problem
"

# ╔═╡ 4be58ce5-6c41-446d-b5a2-8e47d84c4569
begin
	Vₜ = Array{Float32}(undef, n₁, n₂, n₂)

	for j ∈ axes(Vₜ, 2)
		Vₜ[:, j, :] .= V̄
	end

	∇Vₜ = Utils.central∇(Vₜ, Ω)
end;

# ╔═╡ ef2c5271-40d0-4226-8088-770853122a3e
t = 10f0;

# ╔═╡ 36782b8c-9802-48c6-a621-528db54b535c
function objective(χ, α, Xᵢ, Vᵢ, ∇Vᵢ)
	Model.f(χ, Xᵢ[3], Vᵢ[1], economy) + 
        ∇Vᵢ[2] * (Model.γ(t, economy, calibration) - α) + 
        ∇Vᵢ[3] * (
			Model.ϕ(t, χ, economy) - 
            Model.A(t, economy) * 
				Model.β(t, Model.ε(t, exp(Xᵢ[2]), α, instance, calibration), economy)
        )
end;

# ╔═╡ d202dcee-1a0f-4f18-a355-bed3c5db4bef
md"
- ``T_i = `` $(@bind Tᵢ Slider(Ω[1], show_value = true, default = hogg.T₀))
- ``m_i = `` $(@bind mᵢ Slider(Ω[2], show_value = true, default = log(hogg.M₀)))
- ``y_i = `` $(@bind yᵢ Slider(Ω[3], show_value = true, default = log(economy.Y₀)))
"

# ╔═╡ 53b622db-b269-4c82-98ac-bcde2d576af4
begin
	i = findfirst(x -> Tᵢ == x, Ω[1]) 
	j = findfirst(x -> mᵢ == x, Ω[2]) 
	k = findfirst(x -> yᵢ == x, Ω[3]) 

	idx = CartesianIndex(i, j, k)

	Xᵢ = [Tᵢ, mᵢ, yᵢ]
	∇Vᵢ = @view ∇Vₜ[idx, 1:3]
	Vᵢ = @view Vₜ[idx]
end;

# ╔═╡ 885b4f32-d4bd-44e5-bb63-a1dd8ee9a11e
begin
	M̄ = [Model.Mstable(T, hogg, albedo) for T ∈ Ω[1]];
end

# ╔═╡ 803b44e3-c482-4aa6-9569-4396c1fb17c0
let
	unit = range(0f0, 1f0; step = 0.01f0)

	pos = (x -> round(Float64(x), digits = 2)).([Tᵢ, exp(mᵢ), exp(yᵢ)])

	objfig = contourf(
		unit, unit, (χ, α) -> objective(χ, α, Xᵢ, Vᵢ, ∇Vᵢ), 
		cmap = :coolwarm, linewidth = 0, aspect_ratio = 1,
		xlabel = "\$\\chi\$", xlims = (0, 1),
		ylabel = "\$\\alpha\$", ylims = (0, 1),
		title = "Objective with \$(T_i, M_i, Y_i) = $pos\$",
		camera = (42, 36), levels = 30, legend = false
	)

	climatefig = plot(M̄, Ω[1])
end

# ╔═╡ 3e1a79b1-b4a3-4781-a6f6-dda13e1f4e75
Model.Mstable(0.5)

# ╔═╡ Cell order:
# ╠═ed817ffc-f1f4-423c-8374-975e34d449eb
# ╠═f3d4f91d-ebac-43cb-9789-df38f9a87a8c
# ╠═e11b7094-b8d0-458c-9e4a-8ebe3b588dfc
# ╠═e74335e3-a230-4d11-8aad-3323961801aa
# ╠═713208a4-2d81-4db5-afd4-9b99816518dc
# ╠═b3ac7b02-2aa5-4da9-9459-0ea24d703697
# ╟─3c65c27d-ee8f-45cf-81ec-5183e1949a59
# ╟─7ca9d16e-a543-429e-b126-29db7bb51796
# ╟─2dc66eff-a2e5-4da9-909f-f95fb5f2b6b9
# ╠═4607f0c4-027b-4402-a8b1-f98750696b6f
# ╠═5d878f4a-279e-424f-8da5-d9de5e491a8d
# ╠═002cf9d6-369a-48e2-8797-31f251637866
# ╠═ddea2917-4271-4c72-b9f0-4e1aa6be42de
# ╠═a948f29f-4ad7-4f59-a624-6dae9c5c2f42
# ╠═a85e9db2-88eb-409b-b44f-9e7c8820e1b9
# ╠═5ba540ee-6ac3-4265-82bd-1a9a840bc927
# ╠═dc2cd7ba-63ef-44aa-b9f4-e6bd003b8cf3
# ╠═f08b07ce-c488-4ff7-adb9-18752a0a3a2e
# ╠═51512c63-296d-4f4a-aa97-97e91df1304f
# ╠═51700462-1e9f-4548-aa07-d82f0fcd20cb
# ╟─f49ffc11-6204-4bdc-a17f-80c4f7e8cdce
# ╠═7d0297b2-a0e7-45b6-b573-cb6015b1c283
# ╠═4be58ce5-6c41-446d-b5a2-8e47d84c4569
# ╠═36782b8c-9802-48c6-a621-528db54b535c
# ╠═ef2c5271-40d0-4226-8088-770853122a3e
# ╠═53b622db-b269-4c82-98ac-bcde2d576af4
# ╟─d202dcee-1a0f-4f18-a355-bed3c5db4bef
# ╠═885b4f32-d4bd-44e5-bb63-a1dd8ee9a11e
# ╠═803b44e3-c482-4aa6-9569-4396c1fb17c0
# ╠═3e1a79b1-b4a3-4781-a6f6-dda13e1f4e75
