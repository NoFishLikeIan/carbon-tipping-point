### A Pluto.jl notebook ###
# v0.19.25

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

# ╔═╡ e74335e3-a230-4d11-8aad-3323961801aa
default(size = 500 .* (√2, 1), dpi = 180, titlefontsize = 12)

# ╔═╡ 3c65c27d-ee8f-45cf-81ec-5183e1949a59
md"## Epstein-Zin"

# ╔═╡ 2dc66eff-a2e5-4da9-909f-f95fb5f2b6b9
md"## Import terminal condition"

# ╔═╡ 4607f0c4-027b-4402-a8b1-f98750696b6f
datapath = joinpath("..", get(DotEnv.config(), "DATAPATH", "data/"), "terminal.jld2") 

# ╔═╡ 002cf9d6-369a-48e2-8797-31f251637866
@load datapath V̄;

# ╔═╡ ddea2917-4271-4c72-b9f0-4e1aa6be42de
economy = Model.Economy(); hogg = Model.Hogg(); albedo = Model.Albedo();

# ╔═╡ 7ca9d16e-a543-429e-b126-29db7bb51796
begin
	consumption = range(0, economy.Ȳ; length = 101)
	utility = range(-2, 0; length = 101)
	
	contourf(consumption, utility, (c, u) -> Model.f(c, 0, u, economy))
end

# ╔═╡ 9da9c5e5-f2c7-4be8-bdc4-adf3523d8e8e
size(V̄)

# ╔═╡ a85e9db2-88eb-409b-b44f-9e7c8820e1b9
begin
	n₁, n₂ = size(V̄)
	domain = [(hogg.T₀, hogg.T̄, n₁), (log(economy.Y̲), log(economy.Ȳ), n₂)]
	Ω = Utils.makegrid(domain)
end;

# ╔═╡ bb5ef699-15c5-4713-88ef-14c67abc59bf
md"Camera: $(@bind αₓ Slider(-90:1:90, default = 38, show_value = true))"

# ╔═╡ 51512c63-296d-4f4a-aa97-97e91df1304f
let
	surface(Ω[1], Ω[2], V̄'; 
		xlabel = "\$T\$", ylabel = "\$y\$", title = "\$\\bar{V}(T, y)\$",
		legend = false, camera = (αₓ, 30)
	)
end

# ╔═╡ Cell order:
# ╠═ed817ffc-f1f4-423c-8374-975e34d449eb
# ╠═f3d4f91d-ebac-43cb-9789-df38f9a87a8c
# ╠═e11b7094-b8d0-458c-9e4a-8ebe3b588dfc
# ╠═e74335e3-a230-4d11-8aad-3323961801aa
# ╟─3c65c27d-ee8f-45cf-81ec-5183e1949a59
# ╠═7ca9d16e-a543-429e-b126-29db7bb51796
# ╟─2dc66eff-a2e5-4da9-909f-f95fb5f2b6b9
# ╟─4607f0c4-027b-4402-a8b1-f98750696b6f
# ╠═002cf9d6-369a-48e2-8797-31f251637866
# ╠═ddea2917-4271-4c72-b9f0-4e1aa6be42de
# ╠═9da9c5e5-f2c7-4be8-bdc4-adf3523d8e8e
# ╠═a85e9db2-88eb-409b-b44f-9e7c8820e1b9
# ╟─bb5ef699-15c5-4713-88ef-14c67abc59bf
# ╠═51512c63-296d-4f4a-aa97-97e91df1304f
