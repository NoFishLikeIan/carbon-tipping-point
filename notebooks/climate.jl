### A Pluto.jl notebook ###
# v0.19.30

using Markdown
using InteractiveUtils

# ╔═╡ b9cc9ff6-7fc8-11ee-3f84-d90577b8ac4a
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(Base.current_project())
    Pkg.instantiate()

	using UnPack, JLD2, DotEnv
end

# ╔═╡ 70b25c8f-b704-4eb3-af84-16bdf7119c90
using Roots

# ╔═╡ 1471d3ee-6830-4955-9ba6-e5c004701794
using Model, Utils

# ╔═╡ aef78453-73c2-43b9-ad59-979574471c4d
using Plots, PlutoUI; default(size = 500 .* (√2, 1), dpi = 180, titlefontsize = 12)

# ╔═╡ 24c016a9-1628-46f4-9a6f-0a6c34de2d7e
TableOfContents()

# ╔═╡ 1f4dd3bb-5350-43e8-a6e5-216f698fd4e7
begin
	hogg = Hogg(σ²ₜ = 0.01f0)
	albedo = Albedo(λ₂ = Albedo().λ₁ - 0.09)

	@unpack S₀, η = hogg

	Tᵢ = (albedo.T₁ + albedo.T₂) / 2f0 
	c = Model.secondstoyears / hogg.ϵ
end;

# ╔═╡ b0c570e4-9c92-476c-b842-c81181f5598d
Tstable(M, albedo) = find_zeros(T -> Model.Mstable(T, hogg, albedo) - M, 288f0, 297f0);

# ╔═╡ 698fd792-b847-49e2-871d-bcb558461b5e
begin
	n = 101
	Tspace = range(hogg.Tᵖ, hogg.Tᵖ + 10f0; length = n)
	Tspaceext = range(200f0, 400f0; length = 3n)

	Mspace = range(hogg.Mᵖ, 4hogg.Mᵖ; length = 2n)
	mspace = log.(Mspace)
end;

# ╔═╡ 517a32a7-098d-4c89-b428-eed70c41e1d4
function V(T, m, albedo::Albedo)
	@unpack λ₁, λ₂ = albedo
	G = Model.fₘ(m, hogg)

	(η / 5f0) * T^5 - G * T - (1 - λ₁) * S₀ * T - S₀ * (λ₁ - λ₂) * log(1 + exp(T - Tᵢ))
end;

# ╔═╡ a0a13647-a2af-4ad9-8eb6-c20c1d18d7bb
let
	M = hogg.M₀
		
	plot(Tspace, T -> V(T, log(1.2M), Albedo(λ₂ = 0.20)); c = :darkred, linewidth = 2, label = "\$V(t)\$")
	vline!([hogg.T₀]; c = :black, linestyle = :dashdot, label = "\$T_0\$")
end

# ╔═╡ 6ecc6397-8a26-4a7c-8145-4415ba6bf94d
hogg.Tᵖ + 1.4

# ╔═╡ 3be104fd-a0ae-4eb5-88f8-04aa8b9a48b1
function p(T, m, albedo; Vz = 10f-4)
	exp(-2(V(T, m, albedo) * Vz) / (hogg.σ²ₜ / c^2))
end;

# ╔═╡ aeda50d8-02e1-49af-b814-adc59a3d7a8a
begin
	p̂ = [p(T, log(M), albedo) for T ∈ Tspace, M ∈ Mspace]
	p̂ = p̂ ./ sum(p̂, dims = 1)
end;

# ╔═╡ 762208c8-9027-4eb2-bbd5-f62493d1b19d
let
	nullcline = [Model.Mstable(T, hogg, albedo) for T ∈ Tspace]

	heatmap(Mspace, Tspace, p̂, linewidth = 0, c = :Blues, cbar = false)
	plot!(nullcline, Tspace; xlims = extrema(Mspace), ylims = extrema(Tspace), linewidth = 3, linestyle = :dash, c = :white, label = false)


end

# ╔═╡ 29c82a92-1423-4ca1-aca1-50cd15d864ee
plot(p̂[:, 200])

# ╔═╡ 768d2952-598d-48a4-ab02-39075f1fdd21
sum(p̂[:, 200])

# ╔═╡ Cell order:
# ╠═b9cc9ff6-7fc8-11ee-3f84-d90577b8ac4a
# ╠═70b25c8f-b704-4eb3-af84-16bdf7119c90
# ╠═1471d3ee-6830-4955-9ba6-e5c004701794
# ╠═aef78453-73c2-43b9-ad59-979574471c4d
# ╠═24c016a9-1628-46f4-9a6f-0a6c34de2d7e
# ╠═1f4dd3bb-5350-43e8-a6e5-216f698fd4e7
# ╠═b0c570e4-9c92-476c-b842-c81181f5598d
# ╠═698fd792-b847-49e2-871d-bcb558461b5e
# ╠═517a32a7-098d-4c89-b428-eed70c41e1d4
# ╠═a0a13647-a2af-4ad9-8eb6-c20c1d18d7bb
# ╠═6ecc6397-8a26-4a7c-8145-4415ba6bf94d
# ╠═3be104fd-a0ae-4eb5-88f8-04aa8b9a48b1
# ╠═aeda50d8-02e1-49af-b814-adc59a3d7a8a
# ╠═762208c8-9027-4eb2-bbd5-f62493d1b19d
# ╠═29c82a92-1423-4ca1-aca1-50cd15d864ee
# ╠═768d2952-598d-48a4-ab02-39075f1fdd21
