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
using Random; rng = MersenneTwister(123);

# ╔═╡ e11b7094-b8d0-458c-9e4a-8ebe3b588dfc
using Plots, PlutoUI

# ╔═╡ 9a954586-c41b-44bb-917e-2262c00b958a
using Roots, Interpolations

# ╔═╡ a1f3534e-8c07-42b1-80ac-440bc016a652
using DifferentialEquations

# ╔═╡ d04d558a-c152-43a1-8668-ab3b040e6701
using DifferentialEquations: EnsembleAnalysis, EnsembleDistributed

# ╔═╡ 93709bdd-408f-4f87-b0c8-fda34b06af57
include("../scripts/plotutils.jl")

# ╔═╡ b29e58b6-dda0-4da9-b85d-d8d7c6472155
TableOfContents()

# ╔═╡ e74335e3-a230-4d11-8aad-3323961801aa
default(size = 500 .* (√2, 1), dpi = 180, titlefontsize = 12)

# ╔═╡ c25e9def-d474-4e97-8919-c066bb11338c
md"
## Parameters

``\Delta\lambda =`` $(@bind Δλ Slider([1e-5, 0.01, 0.08, 0.1], show_value = true, default = 0.08))
"

# ╔═╡ 94be80bf-ebc1-42ed-903d-071361249222
N = 50;

# ╔═╡ c9396e59-ed2f-4f73-bf48-e94ccf6e55bd
md"""
# Post-transition phase
"""

# ╔═╡ 4607f0c4-027b-4402-a8b1-f98750696b6f
begin
	env = DotEnv.config()
	datapath = joinpath("..", get(env, "DATAPATH", "data/"))
	termpath = joinpath(datapath, "terminal", "N=$(N)_Δλ=$(Δλ).jld2")

	V̄ = load(termpath, "V̄")
	model = load(termpath, "model")
	termpolicy = load(termpath, "policy")

	@unpack economy, hogg, albedo, grid, calibration = model
end;

# ╔═╡ 35f6e02d-70cb-4111-8e33-e43c8db5e7a8
X₀ = Point(hogg.T₀, log(hogg.M₀), log(economy.Y₀));

# ╔═╡ c5f9e376-6ab9-4a4f-960d-7dcaf8d03fb6
md"
``m``: $(@bind mtermfig Slider(range(model.grid.domains[2]...; length = 101), show_value = true, default = X₀.m))
"

# ╔═╡ 98749904-9225-4e92-913e-b084eeba4fd7
plotsection(V̄, mtermfig, model; zdim = 2, title = "\$\\overline{V}\$", surf = true, c = :viridis, zlims = (minimum(V̄), 0.), xflip = true)

# ╔═╡ 6fe67c9b-fe20-42e4-b817-b31dad586e55
md"# Backward Simulation"

# ╔═╡ fc2c9720-3607-4ee2-a48c-f8e22d4404bd
md"## Constructing interpolations"

# ╔═╡ 9bcfa0d7-0442-40f0-b63f-7d39f38c1310
# ╠═╡ disabled = true
#=╠═╡
begin
	simpath = joinpath(datapath, "total", "N=50_Δλ=$(Δλ).jld2")
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
  ╠═╡ =#

# ╔═╡ d2c83cdf-002a-47dc-81f9-22b76f183587
#=╠═╡
begin
	ΔT, Δm, Δy = grid.domains

	nodes = (
		range(ΔT[1], ΔT[2]; length = size(grid, 1)),
		range(Δm[1], Δm[2]; length = size(grid, 2)),
		range(Δy[1], Δy[2]; length = size(grid, 3)),
		0:(size(V, 4) - 1)
	)

	χitp = linear_interpolation(nodes, first.(policy); extrapolation_bc = Flat())
	αitp = linear_interpolation(nodes, last.(policy); extrapolation_bc = Flat())
end;
  ╠═╡ =#

# ╔═╡ ffa267c0-4932-4889-a555-a2d07b58344f
md"## Policy"

# ╔═╡ 31d32b62-4329-464e-8354-1c2875fe5801
md"## Simulation"

# ╔═╡ d62b65f5-220e-45c6-a434-ac392b72ab4a
#=╠═╡
function F!(dx, x, p, t)	
	T, m, y = x
	
	χ = χitp(T, m, y, t)
	α = αitp(T, m, y, t)
	
	dx[1] = μ(T, m, hogg, albedo) / hogg.ϵ
	dx[2] = γ(t, economy, calibration) - α
	dx[3] = b(t, x, χ, α, model)

	return
end;
  ╠═╡ =#

# ╔═╡ 7823bda7-5ab8-42f7-bf1c-292dbfecf178
function G!(dx, x, p, t)
	dx[1] = hogg.σₜ / hogg.ϵ
	dx[2] = 0.
	dx[3] = economy.σₖ
	
	return
end;

# ╔═╡ c3c2cfc9-e7f4-495b-bcc6-51227be2c6b5
#=╠═╡
begin
	x₀ = [hogg.T₀, log(hogg.M₀), log(economy.Y₀)]
	prob = SDEProblem(SDEFunction(F!, G!), G!, x₀, (0., 90.))
end
  ╠═╡ =#

# ╔═╡ 28c6fe28-bd42-4aba-b403-b2b0145a8e37
#=╠═╡
begin
	ensembleprob = EnsembleProblem(prob)
	solution = solve(ensembleprob, EnsembleDistributed(); trajectories = 100)
end;
  ╠═╡ =#

# ╔═╡ 1a19b769-68e2-411b-afe0-6bd2a7fb87a3
#=╠═╡
begin
	time = range(prob.tspan...; length = 101)
	median = [Point(EnsembleAnalysis.timepoint_median(solution, t)) for t in time]

	Tfig = plot(time, [x.T for x ∈ median], ylabel = "\$T\$", label = false, linewidth = 3, c = :black)
	Yfig = plot(time, [exp(x.y) for x ∈ median], ylabel = "\$Y\$", xlabel = "\$t\$", label = false, c = :black, linewidth = 3)
	
	for sim in solution
		data = Point.(sim.(time))
		plot!(Tfig, time, [ x.T for x ∈ data ]; label = false, alpha = 0.05, c = :black)
		plot!(Yfig, time, [ exp(x.y) for x ∈ data ]; label = false, alpha = 0.05, c = :black)
	end

	plot(Tfig, Yfig, sharex = true, layout = (2, 1), link = :x)
end
  ╠═╡ =#

# ╔═╡ 43bc8d15-40d5-457c-84f9-57826cb4139f
#=╠═╡
begin
	αfig = plot(ylabel = "\$\\alpha\$")
	χfig = plot(ylabel = "\$\\chi\$")
	
	for sim in solution
		data = Point.(sim.(time))
		plot!(αfig, time, [αitp(time[i], x...) for (i, x) ∈ enumerate(data)]; label = false, alpha = 0.1, c = :darkred)
		plot!(χfig, time, [χitp(time[i], x...) for (i, x) ∈ enumerate(data)]; label = false, alpha = 0.1, c = :darkblue)
	end
	
	plot(αfig, χfig, layout = (2, 1), link = :x)

end
  ╠═╡ =#

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
# ╟─c25e9def-d474-4e97-8919-c066bb11338c
# ╠═94be80bf-ebc1-42ed-903d-071361249222
# ╟─c9396e59-ed2f-4f73-bf48-e94ccf6e55bd
# ╠═4607f0c4-027b-4402-a8b1-f98750696b6f
# ╠═35f6e02d-70cb-4111-8e33-e43c8db5e7a8
# ╟─c5f9e376-6ab9-4a4f-960d-7dcaf8d03fb6
# ╟─98749904-9225-4e92-913e-b084eeba4fd7
# ╟─6fe67c9b-fe20-42e4-b817-b31dad586e55
# ╠═a1f3534e-8c07-42b1-80ac-440bc016a652
# ╠═d04d558a-c152-43a1-8668-ab3b040e6701
# ╟─fc2c9720-3607-4ee2-a48c-f8e22d4404bd
# ╠═9bcfa0d7-0442-40f0-b63f-7d39f38c1310
# ╠═d2c83cdf-002a-47dc-81f9-22b76f183587
# ╟─ffa267c0-4932-4889-a555-a2d07b58344f
# ╟─31d32b62-4329-464e-8354-1c2875fe5801
# ╠═d62b65f5-220e-45c6-a434-ac392b72ab4a
# ╠═7823bda7-5ab8-42f7-bf1c-292dbfecf178
# ╠═c3c2cfc9-e7f4-495b-bcc6-51227be2c6b5
# ╠═28c6fe28-bd42-4aba-b403-b2b0145a8e37
# ╠═1a19b769-68e2-411b-afe0-6bd2a7fb87a3
# ╠═43bc8d15-40d5-457c-84f9-57826cb4139f
