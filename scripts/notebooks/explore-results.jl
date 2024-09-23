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

# ╔═╡ 6d43d072-4b34-11ef-3e45-8fde68196131
begin # Use local module
	import Pkg
  	Pkg.activate(Base.current_project())
    Pkg.instantiate()
end

# ╔═╡ b447bdc5-ea85-462c-a1f0-c8c8d5a3f1e9
using PlutoUI; TableOfContents()

# ╔═╡ 5eb83b47-5f48-463d-8abb-21eaf36dbc25
using JLD2, DotEnv, UnPack

# ╔═╡ 9ca8ccf3-c770-4d13-8ed5-ece9eaaf4b28
using FastClosures, Interpolations

# ╔═╡ cbd38547-eea1-4214-88c7-944c1aca82c2
using Model, Grid

# ╔═╡ f6bab433-5d9d-4516-9c24-59c5441c06eb
begin
	using Plots
	using Printf, LaTeXStrings 

	default(size = 500 .* (√2, 1), dpi = 180, linewidth = 2.5, label = false)
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
begin # Global variables
    env = DotEnv.config("../.env")
    BASELINE_YEAR = 2020
	ALLOWNEGATIVE = false

    DATAPATH = joinpath("..", get(env, "DATAPATH", "data"))
    
    datapath = joinpath(
		DATAPATH, 
		get(env, "SIMULATIONPATH", "simulaton"), 
		ALLOWNEGATIVE ? "negative" : ""
	)

    PLOTPATH = get(env, "PLOTPATH", "plots")
    PRESENTATIONPATH = joinpath(PLOTPATH, "presentation")

    SAVEFIG = false
    SEED = 11148705
end;

# ╔═╡ 542e01e0-bbce-462b-9c43-6d375adb6bd5
begin
	Saving = ingredients("../scripts/utils/saving.jl")
	Simulating = ingredients("../scripts/utils/simulating.jl")
	Plotting = ingredients("../scripts/plotting/utils.jl")
end;

# ╔═╡ abc6d59d-6b20-4193-ab50-53facae969e1
begin # Default parameters
	# Construct models and grids
    thresholds = [1.5, 2.5];

	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	preferences = EpsteinZin()
    damages = GrowthDamages()
    economy = Economy()
    jump = Jump()
    hogg = Hogg()
    
	models = AbstractModel[]

	for Tᶜ ∈ thresholds
	    albedo = Albedo(Tᶜ)
	    model = TippingModel(albedo, hogg, preferences, damages, economy, calibration)

		push!(models, model)

	end

    jumpmodel = JumpModel(jump,  hogg, preferences, damages, economy, calibration)
    push!(models, jumpmodel)
end;

# ╔═╡ cfad70a3-73d0-4d2f-9180-3f2c6322b17d
md"# Set-up"

# ╔═╡ 891cac99-6427-47f2-8956-d4eb7817ea54
begin
    results = Saving.loadtotal.(models; datapath);
    itps = Simulating.buildinterpolations.(results);

	resultsmap = Dict{AbstractModel, typeof(first(results))}(models .=> results)
    itpsmap = Dict{AbstractModel, typeof(first(itps))}(models .=> itps);
    abatementmap = Dict{AbstractModel, Interpolations.Extrapolation}(model => itp[:α] for (model, itp) in itpsmap)

	G = last(results[1])

	Tspace = range(G.domains[1]...; length = size(G, 1))
	mspace = range(G.domains[2]...; length = size(G, 2))

	labels = Dict{AbstractModel, String}(models .=> ["imminent", "remote", "benchmark"])
end;

# ╔═╡ 78c11f3a-1325-46e5-974b-ae67b9637834
md"
# Results of simulation

Scenario: $( @bind plotmodel (labels |> collect |> Select) )
"

# ╔═╡ d7df916d-cf45-4564-86ad-5b64e309cb11
begin
	result = resultsmap[plotmodel]
	itp = itpsmap[plotmodel]
end;

# ╔═╡ 80bfe18d-8d06-4957-a4ad-dd80bf8c42b1
md"
## Policy

`t =` $(@bind tfigpol Slider(range(extrema(first(result))...; step = 1/4), default = 0., show_value = true))
"

# ╔═╡ 033b0310-2df1-4613-81f8-39274d2318ba
let
	@unpack α, χ = itp
	
	χₜ = @closure (m, T) -> χ(T, m, tfigpol)
	dm = @closure (m, T) -> 1 - ε(tfigpol, exp(m), α(T, m, tfigpol), plotmodel)

	nullcline = [mstable(T, plotmodel) for T in Tspace]
	
	consfig = heatmap(mspace, Tspace, χₜ; ylabel = "\$T\$", xlabel = "\$m\$", title = "Time \$t = $tfigpol\$; \$\\chi\$", xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)
	plot!(consfig, nullcline, Tspace; linestyle = :dash, c = :white, label = false, linewidth = 3)

	cmap = ALLOWNEGATIVE ? :coolwarm : :coolwarm
	clims = ALLOWNEGATIVE ? (-1., 1.) : (-1., 1.)

	abatfig = heatmap(mspace, Tspace, dm; ylabel = "\$T\$", xlabel = "\$m\$", title = "\$E / E^b\$", xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0., c = :coolwarm, clims = (-1., 1.))
	plot!(abatfig , nullcline, Tspace; linestyle = :dash, c = :white, label = false, linewidth = 3)

	plot(consfig, abatfig; size = 500 .* (2√2, 1), margins = 10Plots.mm)
end

# ╔═╡ 28c7e9d4-d34a-4bcb-87a2-1c568ec8c669
md"
## Value

`t =` $(@bind tfigval Slider(range(extrema(first(result))...; step = 1/4), default = 0., show_value = true))
"

# ╔═╡ 78a476ce-788c-428d-b6bf-da39f92f4035
let
	@unpack F = itp
	
	Fₜ = @closure (m, T) -> log(F(T, m, tfigval))

	nullcline = [mstable(T, plotmodel) for T in Tspace]
	
	consfig = contourf(mspace, Tspace, Fₜ; ylabel = "\$T\$", xlabel = "\$m\$", title = L"Time $t = %$(tfigval)$; $\log F_t(T, m)$", xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0., clims = (-10., 20.), c = :Reds)
	plot!(consfig, nullcline, Tspace; linestyle = :dash, c = :white, label = false, linewidth = 3)
end

# ╔═╡ 497f0b6b-e7ac-4176-81b3-f8df4050d338
md"# Simulation"

# ╔═╡ baa07798-cd01-49cb-8d33-eafd0c92505b
begin # Solve an ensemble problem for all models with the bau scenario
    trajectories = 1_000
    ratejump = VariableRateJump(Simulating.rate, Simulating.tippingopt!);
    sols = Dict{AbstractModel, EnsembleSolution}()

	x₀ = [hogg.T₀, log(hogg.M₀)]

    for model in models
        itp = itpsmap[model];
        αitp = itp[:α];

        prob = SDEProblem(Simulating.F!, Simulating.G!, x₀, (0., 80.), (model, αitp))

        if isa(model, JumpModel)
            jumpprob = JumpProblem(prob, ratejump)
            ensprob = EnsembleProblem(jumpprob)
            abatedsol = solve(ensprob, SRIW1(); trajectories)
        else
            ensprob = EnsembleProblem(prob)
            abatedsol = solve(ensprob; trajectories)
        end

        sols[model] = abatedsol
    end
end;

# ╔═╡ 963b5d26-1f93-4cd9-8a9b-989e22c16143
begin
	yearlytime = 0:80
    yearticks = 0:20:80

    βextrema = (0., 0.035)
    βticks = range(βextrema...; step = 0.01) |> collect
    βticklabels = [@sprintf("%.0f %%", 100 * y) for y in βticks]

    Textrema = (1., 2.1)
    Tticks = Plotting.makedeviationtickz(Textrema..., first(models); step = 0.5, digits = 1)

	simfigs = []
	
    for (k, model) in enumerate(models)
		abatedsol = sols[model]
		itp = itpsmap[model]
		αitp = itp[:α]
	
		# Abatement expenditure figure
		βM = Simulating.computeonsim(abatedsol, (T, m, t) -> β(t, ε(t, exp(m), αitp(T, m, t), model), model.economy), yearlytime)
	   
		βquantiles = Simulating.timequantiles(βM, [0.05, 0.5, 0.95])
		Simulating.smoothquantile!.(eachcol(βquantiles), 10)

       	βoptionfirst = k > 1 ? Dict() : Dict(:title => L"Abatement as % of $Y_t$" )
       	lastxticks = Dict(
			:xticks => k < length(models) ? (yearticks, repeat([""], length(yearticks))) : (yearticks, BASELINE_YEAR .+ yearticks)
		)

		βfig = plot(yearlytime, βquantiles[:, 2]; c = :darkgreen, ylims = βextrema, yticks = (βticks, βticklabels), βoptionfirst..., lastxticks...)
		plot!(βfig, yearlytime, βquantiles[:, 1]; fillrange = βquantiles[:, 3], alpha = 0.3, c = :darkgreen, linewidth = 0.)

       	push!(simfigs, βfig)

		# Temperature figure
		paths = EnsembleAnalysis.timeseries_point_quantile(abatedsol, [0.05, 0.5, 0.95], yearlytime)
		Tpaths = first.(paths.u)

		Toptionfirst = k > 1 ? Dict() : Dict(:title => L"Temperature $T_t$" )
		Tfig = plot(yearlytime, getindex.(Tpaths, 2); c = :darkred, yticks = Tticks, ylims = Textrema .+ hogg.Tᵖ, Toptionfirst..., lastxticks...)
		plot!(Tfig, yearlytime, getindex.(Tpaths, 1); fillrange = getindex.(Tpaths, 3), alpha = 0.3, c = :darkred, linewidth = 0.)


		push!(simfigs, Tfig)
    end;

    plot(simfigs...; link = :x, layout = (3, 2), size = 300 .* (2√2, 2))
end

# ╔═╡ 32955af4-fa47-43dd-b6a8-7545e1398f25
md"# Regret"

# ╔═╡ 39409585-662d-4915-ae42-653e35a2b975
begin # Solve the regret problem. Discover tipping point only after T ≥ Tᶜ.
    modelimminent, modelremote = models[[1, 2]]
    αimminent = abatementmap[modelimminent]
    αremote = abatementmap[modelremote]

    initparams = (modelimminent, αremote)

	function hittipping(u, t, integrator)
    	model, α = integrator.p
        Tupper = model.albedo.Tᶜ + model.hogg.Tᵖ + (model.albedo.ΔT / 2)

		ΔT = Tupper - u[1]
    	
		return ΔT
    end

	function changepolicy!(integrator)
        integrator.p = (modelimminent, αimminent)
    end

	callback = ContinuousCallback(hittipping, changepolicy!);
    regretprob = SDEProblem(Simulating.F!, Simulating.G!, x₀, (0., 80.), initparams) |> EnsembleProblem

    regretsol = solve(regretprob; trajectories, callback)
end;

# ╔═╡ 9283bc5d-f5be-4b39-bc02-aeb942f1db79
plot(regretsol; idxs = 1, linewidth = 0.5, c = :black, opacity = 0.1)

# ╔═╡ 68d51817-7f7e-4ebb-b986-67280479a999
plot(regretsol; idxs = 2, linewidth = 0.5, c = :black, opacity = 0.1)

# ╔═╡ 498bf7bf-4812-4525-864e-db5454c53211
βregret = @closure (T, m, t) -> begin
	model = ifelse(T - hogg.Tᵖ > modelimminent.albedo.Tᶜ, modelimminent, modelremote)

	αitp = abatementmap[model]

	return β(t, ε(t, exp(m), αitp(T, m, t), model), model.economy)
end;

# ╔═╡ f5584590-689c-4ddb-8e64-40c5c22e34ed
βM = Simulating.computeonsim(regretsol, βregret, yearlytime)

# ╔═╡ 9c1452bc-611f-4e17-a9f5-89a80acf9568
let
	βM = Simulating.computeonsim(regretsol, βregret, yearlytime)
	βquantiles = Simulating.timequantiles(βM, [0.1, 0.5, 0.9])
    Simulating.smoothquantile!.(eachcol(βquantiles), 0)

	regretfig = plot(yearlytime, βquantiles[:, 2])
end

# ╔═╡ Cell order:
# ╟─35f99656-f68a-455c-9042-b5afc5e7a4a8
# ╟─44eee19a-595a-42fe-9302-0f19df42388a
# ╟─6d43d072-4b34-11ef-3e45-8fde68196131
# ╠═b447bdc5-ea85-462c-a1f0-c8c8d5a3f1e9
# ╟─729c4427-986c-48b2-ae05-3b7b51c7d6bd
# ╠═5eb83b47-5f48-463d-8abb-21eaf36dbc25
# ╠═9ca8ccf3-c770-4d13-8ed5-ece9eaaf4b28
# ╠═576db675-6ca4-4f1e-8782-5bc39ef335e6
# ╠═cbd38547-eea1-4214-88c7-944c1aca82c2
# ╠═f6bab433-5d9d-4516-9c24-59c5441c06eb
# ╠═bfa6364c-7376-449e-90bc-11a37334092a
# ╠═542e01e0-bbce-462b-9c43-6d375adb6bd5
# ╠═abc6d59d-6b20-4193-ab50-53facae969e1
# ╟─cfad70a3-73d0-4d2f-9180-3f2c6322b17d
# ╠═891cac99-6427-47f2-8956-d4eb7817ea54
# ╟─78c11f3a-1325-46e5-974b-ae67b9637834
# ╟─d7df916d-cf45-4564-86ad-5b64e309cb11
# ╟─80bfe18d-8d06-4957-a4ad-dd80bf8c42b1
# ╠═033b0310-2df1-4613-81f8-39274d2318ba
# ╟─28c7e9d4-d34a-4bcb-87a2-1c568ec8c669
# ╟─78a476ce-788c-428d-b6bf-da39f92f4035
# ╟─497f0b6b-e7ac-4176-81b3-f8df4050d338
# ╠═baa07798-cd01-49cb-8d33-eafd0c92505b
# ╟─963b5d26-1f93-4cd9-8a9b-989e22c16143
# ╟─32955af4-fa47-43dd-b6a8-7545e1398f25
# ╠═39409585-662d-4915-ae42-653e35a2b975
# ╠═9283bc5d-f5be-4b39-bc02-aeb942f1db79
# ╠═68d51817-7f7e-4ebb-b986-67280479a999
# ╠═498bf7bf-4812-4525-864e-db5454c53211
# ╠═f5584590-689c-4ddb-8e64-40c5c22e34ed
# ╠═9c1452bc-611f-4e17-a9f5-89a80acf9568
