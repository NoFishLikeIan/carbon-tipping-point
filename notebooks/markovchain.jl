### A Pluto.jl notebook ###
# v0.19.38

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

# ╔═╡ bfc6af5e-a261-4d7e-9a16-f4ab54c6e1ca
using PlutoUI

# ╔═╡ 2b3c9a7f-8078-415d-b132-999a01aca419
using JLD2, FastClosures

# ╔═╡ 62a3b757-cf0b-4f8d-bdb8-7edc7ba04d47
begin
	using Optimization
	using OptimizationNLopt, OptimizationMultistartOptimization
	using OptimizationOptimJL
end

# ╔═╡ 9655de0d-73f7-4332-85d9-974a73b4fce1
using Model, Grid

# ╔═╡ abb69118-7aeb-4154-9e3c-680b2816953d
using Interpolations: Extrapolation

# ╔═╡ ddd1a388-76fe-482f-ad8e-fbc1096f2d43
begin
	using Plots
	default(size = 500 .* (√2, 1), dpi = 180, linewidth = 2, cmap = :viridis)

	using LaTeXStrings, Printf
end

# ╔═╡ 0e58697d-75fe-4ca8-a464-6c36bf508466
using Dates

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
	Simulating = ingredients("../scripts/utils/simulating.jl")
	Plotting = ingredients("../scripts/plotting/utils.jl")
	Chain = ingredients("../scripts/markov/chain.jl")
	Terminal = ingredients("../scripts/markov/terminal.jl")
	Backward = ingredients("../scripts/markov/backward.jl")
end;

# ╔═╡ 829e9116-83aa-48f2-b0af-003c0d1d063c
md"## Import simulation"

# ╔═╡ a308bb59-cf5f-4d11-b03e-bbe3169f3a71
begin
	ALLOWNEGATIVE = false
	datapath = "../data/simulation-medium"
	filepaths = joinpath(datapath, ALLOWNEGATIVE ? "negative" : "constrained")
	simulationfiles = Saving.listfiles(filepaths)
	simulationfilesnames = @. replace(basename(simulationfiles), ".jld2" => "")
end;

# ╔═╡ 94232fa5-9ab4-49f6-9f8b-d97bd5dbd9c4
md"
# Scenario

$(@bind filepath Select(simulationfiles .=> simulationfilesnames))
"

# ╔═╡ 720e031b-5140-4287-8964-99799317953e
filepath

# ╔═╡ b45c4d5f-d91b-4b07-b629-a20e9d52ca90
begin
   	result = Saving.loadtotal(filepath)
	timesteps, F, _, G, model = result

	itp = Simulating.buildinterpolations(result)
end;

# ╔═╡ 1c80b078-b4ea-47a1-bcc7-063ddb6eb0cb
begin
	Tdomain, mdomain = G.domains
	
	Tspace = range(Tdomain...; length = size(G, 1))
	mspace = range(mdomain...; length = size(G, 2))
end;

# ╔═╡ 28f9e438-f800-4fee-a216-6daa4ead8da6
begin # Plotting utilities
	idxbyspace(T, m) = idxbyspace(Point(T, m))
	
	function idxbyspace(x::Point)
		i = findfirst(≥(x.T), Tspace)
		j = findfirst(≥(x.m), mspace)
		return CartesianIndex(i, j)
	end

	Tdev = extrema(Tdomain) .- model.hogg.Tᵖ

	mticks = range(mdomain...; length=6)
	Mticks = (mticks, round.(exp.(mticks), digits = 2))

	Tticks = Plotting.makedeviationtickz(Tdev..., model; step = 1)
end;

# ╔═╡ 4a5f12f3-6af4-4aaf-8129-c42fe0070e8b
md"Plot time $(@bind t Slider(timesteps, default = 0., show_value = true))"

# ╔═╡ 4f53f52a-f3b7-48fb-b63c-9fbc7f9dd65e
begin
	Ffig = heatmap(
		mspace, Tspace, (m, T) -> log(itp[:F](T, m, t)); 
		xticks = Mticks, yticks = Tticks, ylabel = L"T_t - T^p", xlabel = L"M_t", title = L"$F_{%$t}(T, M)$", 
		xlims = mdomain, ylims = Tdomain, clims = log.(extrema(F)))

	Tdense = range(Tdomain...; length = 101)
	nullcline = mstable.(Tdense, model.hogg, model.albedo)

	plot!(Ffig, nullcline, Tdense; c = :white, label = false)
end

# ╔═╡ 015922f1-c954-42db-887e-4499e5dbca59
md"# Backward costs"

# ╔═╡ 5aa075ef-8ad2-4d52-b4da-d99d687a7d4e
begin
	function objective(u, p::NTuple{3, Float64})
		t, T, m = p
		
		idx = idxbyspace(T, m)
		return objective(u, (t, idx))
	end
	
    function objective(u, p::Tuple{Float64, CartesianIndex})
        t, idx = p

		tdx = findfirst(x -> x ≥ t, timesteps)
		Fₜ = @view F[:, :, tdx]

        Fᵉ, Δt = Chain.markovstep(t, idx, Fₜ, u[2], model, G)
        return Chain.logcost(Fᵉ, t, G.X[idx], Δt, u, model)
    end

    fn = OptimizationFunction(objective, Optimization.AutoForwardDiff())
end;

# ╔═╡ 0169acc9-2d30-4e80-a38b-a0cbc5af15dc
md"
- ``T =`` $(@bind Tfig Slider(Tspace, default = model.hogg.T₀, show_value = true))
- ``m =`` $(@bind mfig Slider(mspace, default = log(model.hogg.M₀), show_value = true))
"

# ╔═╡ ec33fe87-f4fd-4454-bfa7-e644bb89f344
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	p = (t, Tfig, mfig)
	
	ᾱ = γ(t, model.calibration) + δₘ(exp(mfig), model.hogg)
	u₀ = [0.5, ᾱ / 2]

	prob = OptimizationProblem(fn, u₀, p, lb = zeros(2), ub = [1., ᾱ])
    sol = solve(prob, MultistartOptimization.TikTak(100), LBFGS())

	if !SciMLBase.successful_retcode(sol)
		@warn "Optimisation not successful"
	end
end;
  ╠═╡ =#

# ╔═╡ fb36873a-3db7-439e-955f-24e0725bd6b3
#=╠═╡
let
	cspace = range(0.2, 0.8; length = 201)
	aspace = range(0., 0.05; length = 201)

	Fobjfig = deepcopy(Ffig)
	scatter!(Fobjfig, [mfig], [Tfig], c = :white, label = false, markersize = 5)

	objfig = contour(cspace, aspace, (χ, α) -> objective([χ, α], p); 
		ylims = extrema(aspace), xlims = extrema(cspace),
		xlabel = L"\chi", ylabel = L"\alpha", linewidth = 2, cbar = false, levels = 101
	)

	hline!(objfig, [ᾱ]; linestyle = :dash, label = false, color = :black)
	scatter!(objfig, sol.u[[1]], sol.u[[2]]; label = false, c = :green, markerstrokewidth = 0)
	
	plot(objfig, Fobjfig; size = 410 .* (2√2, 1), margins = 5Plots.mm)
end
  ╠═╡ =#

# ╔═╡ bf05d920-7c30-4863-8063-71c38e308e70
begin
	policy = Array{Float64}(undef, size(G)..., 2)

	Base.Threads.@threads for idx in CartesianIndices(G)
		Xᵢ = G.X[idx]
	
		ᾱ = γ(t, model.calibration) + δₘ(exp(Xᵢ.m), model.hogg)

		u₀ = [0.5, ᾱ / 2]

		prob = OptimizationProblem(fn, u₀, (t, idx), lb = zeros(2), ub = [1., ᾱ])
	    sol = solve(prob, MultistartOptimization.TikTak(100), LBFGS())

		policy[idx, :] .= sol.u
	end
end

# ╔═╡ 209947c9-47b3-4010-916d-d281ec125c2e
begin
	α = @view policy[:, :, 2]

	αfig = heatmap(
		mspace, Tspace, α; 
		xticks = Mticks, yticks = Tticks, ylabel = L"T_t - T^p", xlabel = L"M_t", title = L"$\alpha_{%$t}$", 
		xlims = mdomain, ylims = Tdomain, clims = extrema(α), c = :Greens)


	plot!(αfig, nullcline, Tdense; c = :white, label = false)
end

# ╔═╡ bbbff34a-30ab-479c-b916-2c637ecdaa45
extrema(log.(policy[:, :, 2]))

# ╔═╡ Cell order:
# ╟─e29f796c-c57c-40c3-988a-b7d9295c3dac
# ╟─c4befece-4e47-11ef-36e3-a97f8e06d12b
# ╟─fd94250f-950c-4d37-94ad-ca7d73637d08
# ╠═bfc6af5e-a261-4d7e-9a16-f4ab54c6e1ca
# ╠═2b3c9a7f-8078-415d-b132-999a01aca419
# ╠═62a3b757-cf0b-4f8d-bdb8-7edc7ba04d47
# ╠═9655de0d-73f7-4332-85d9-974a73b4fce1
# ╠═abb69118-7aeb-4154-9e3c-680b2816953d
# ╠═ddd1a388-76fe-482f-ad8e-fbc1096f2d43
# ╠═e0877f7e-6247-4c22-a1c7-fdc42ad4dec2
# ╟─829e9116-83aa-48f2-b0af-003c0d1d063c
# ╠═a308bb59-cf5f-4d11-b03e-bbe3169f3a71
# ╟─94232fa5-9ab4-49f6-9f8b-d97bd5dbd9c4
# ╠═720e031b-5140-4287-8964-99799317953e
# ╠═b45c4d5f-d91b-4b07-b629-a20e9d52ca90
# ╠═1c80b078-b4ea-47a1-bcc7-063ddb6eb0cb
# ╠═28f9e438-f800-4fee-a216-6daa4ead8da6
# ╟─4a5f12f3-6af4-4aaf-8129-c42fe0070e8b
# ╟─4f53f52a-f3b7-48fb-b63c-9fbc7f9dd65e
# ╟─015922f1-c954-42db-887e-4499e5dbca59
# ╠═0e58697d-75fe-4ca8-a464-6c36bf508466
# ╠═5aa075ef-8ad2-4d52-b4da-d99d687a7d4e
# ╟─0169acc9-2d30-4e80-a38b-a0cbc5af15dc
# ╠═ec33fe87-f4fd-4454-bfa7-e644bb89f344
# ╟─fb36873a-3db7-439e-955f-24e0725bd6b3
# ╠═bf05d920-7c30-4863-8063-71c38e308e70
# ╠═209947c9-47b3-4010-916d-d281ec125c2e
# ╠═bbbff34a-30ab-479c-b916-2c637ecdaa45
