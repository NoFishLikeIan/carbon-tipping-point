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

# ╔═╡ d8c65298-2446-4135-b2a8-dbaf0aa90660
using Interpolations: Extrapolation

# ╔═╡ 5eb83b47-5f48-463d-8abb-21eaf36dbc25
using JLD2, UnPack

# ╔═╡ 9ca8ccf3-c770-4d13-8ed5-ece9eaaf4b28
using FastClosures

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
md"# Setup
## Load modules
"

# ╔═╡ 542e01e0-bbce-462b-9c43-6d375adb6bd5
begin
	Saving = ingredients("../scripts/utils/saving.jl")
	Simulating = ingredients("../scripts/utils/simulating.jl")
	Plotting = ingredients("../scripts/plotting/utils.jl")
end;

# ╔═╡ a8afb34d-0273-4597-bbac-9650c172ed3b
md"## Import simulation"

# ╔═╡ caba865f-055d-4655-91ca-14c537709dab
begin
	ALLOWNEGATIVE = false
	datapath = "../data/simulation-medium"
	filepaths = joinpath(datapath, ALLOWNEGATIVE ? "negative" : "constrained")
	simulationfiles = Saving.listfiles(filepaths)
	simulationfilesnames = @. replace(basename(simulationfiles), ".jld2" => "")

	itpmap = Dict{AbstractModel, Dict{Symbol, Extrapolation}}();
    models = AbstractModel[];

    for filepath in simulationfiles
		try 
	        result = Saving.loadtotal(filepath)
	        interpolations = Simulating.buildinterpolations(result)
	        model = last(result)
	
	        itpmap[model] = interpolations
	        push!(models, model)
		catch error
			if error isa JLD2.InvalidDataException
				@warn "Invalid data in $filepath: $error"
			else
				rethrow(error)
			end
		end
    end
end;

# ╔═╡ e431c86c-c65e-48b0-ad78-50aeeb5ed63a
begin
	idxs = sortperm(models; by = model -> (model.preferences.ψ, model.preferences.θ))
	sortedmodels = models[idxs]
	sortedsimulationfilesnames = simulationfilesnames[idxs]
end;

# ╔═╡ cfad70a3-73d0-4d2f-9180-3f2c6322b17d
md"
# Scenario

$(@bind model Select(sortedmodels .=> sortedsimulationfilesnames))
"

# ╔═╡ 9b215416-796e-4007-87bc-f6f0ed696b13
begin
	itp = itpmap[model];
	Tspace = range(0., 8.; length = 101) .+ model.hogg.Tᵖ
	mspace = range(log(model.hogg.Mᵖ), log(2.5model.hogg.Mᵖ); length = 101)
end;

# ╔═╡ 80bfe18d-8d06-4957-a4ad-dd80bf8c42b1
md"
## Policy

`t =` $(@bind tfig Slider(0:1/4:80, default = 0., show_value = true))
"

# ╔═╡ 033b0310-2df1-4613-81f8-39274d2318ba
let
	@unpack α, χ = itp
	
	χₜ = @closure (m, T) -> χ(T, m, tfig)
	dm = @closure (m, T) -> ε(tfig, exp(m), α(T, m, tfig), model)

	nullcline = [mstable(T, model) for T in Tspace]
	
	consfig = heatmap(mspace, Tspace, χₜ; ylabel = "\$T\$", xlabel = "\$m\$", title = "Time \$t = $tfig\$; \$\\chi\$", xlims = extrema(mspace), ylims = extrema(Tspace), clims = (0.3, 0.6))
		
	clims = ALLOWNEGATIVE ? (-1., 1.) : (0, 1.)
	cmap = ALLOWNEGATIVE ? :coolwarm : :Greens
	
	abatfig = heatmap(mspace, Tspace, dm; ylabel = "\$T\$", xlabel = "\$m\$", title = "\$\\varepsilon\$", xlims = extrema(mspace), ylims = extrema(Tspace), clims, cmap)

	for fig in (consfig, abatfig)
		plot!(fig, nullcline, Tspace; linestyle = :dash, c = :white, label = false, linewidth = 3)
	end

	# plot(consfig, abatfig; size = 500 .* (2√2, 1), margins = 10Plots.mm)

	abatfig
end

# ╔═╡ e5895749-efc8-410a-bc1a-562979335a0b
md"## Value"

# ╔═╡ 78a476ce-788c-428d-b6bf-da39f92f4035
let
	@unpack F = itp
	
	Fₜ = @closure (m, T) -> log(abs(F(T, m, tfig)))

	nullcline = [mstable(T, model) for T in Tspace]
	
	consfig = contourf(mspace, Tspace, Fₜ; ylabel = "\$T\$", xlabel = "\$m\$", title = L"Time $t = %$(tfig)$; $\log F_t(T, m)$", xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0., c = :Reds)
	plot!(consfig, nullcline, Tspace; linestyle = :dash, c = :white, label = false, linewidth = 3)
end

# ╔═╡ 497f0b6b-e7ac-4176-81b3-f8df4050d338
md"## Simulation"

# ╔═╡ baa07798-cd01-49cb-8d33-eafd0c92505b
begin # Solve an ensemble problem for all models with the bau scenario
    trajectories = 20
    ratejump = VariableRateJump(Simulating.rate, Simulating.tippingopt!);
    sols = Dict{AbstractModel, EnsembleSolution}()

	initialpoints = [[T₀, log(model.hogg.M₀), log(model.economy.Y₀)] for T₀ in Simulating.sampletemperature(model, trajectories)];

	resample = @closure (prob, id, _) -> begin
            if prob isa JumpProblem
                prob.prob.u0[1:3] .= initialpoints[id]
                return prob
            else
                prob.u0 .= initialpoints[id]
                return prob
            end
        end

	interpolations = itpmap[model];
	policies = (interpolations[:χ], interpolations[:α]);
	parameters = (model, policies);

	problem = SDEProblem(Simulating.F!, Simulating.G!, first(initialpoints), (0., 140.), parameters)

	if model isa JumpModel
		ratejump = VariableRateJump(Simulating.rate, Simulating.tippingopt!)
		problem = JumpProblem(problem, ratejump)
	end

	ensembleprob = EnsembleProblem(problem; prob_func = resample)
	simulation = solve(ensembleprob, SRIW1(); trajectories = trajectories)
end;

# ╔═╡ 3bbdb5f8-b258-4b93-b939-cad8d5c5fc11
begin
	βfn = @closure (T, m, y, t) -> begin
            abatement = interpolations[:α](T, m, t)
            emissivity = ε(t, exp(m), abatement, model)
            return β(t, emissivity, model.economy)
        end
	βtime = 0:0.1:80
	βsim = Simulating.computeonsim(simulation, βfn, βtime)

	Tfig = plot(simulation; idxs = 1, linewidth = 1, c = :black, alpha = 0.5, dpi = 180)
	βfig = plot(βtime, βsim; linewidth = 1, c = :black, alpha = 0.5, dpi = 180)	


	plot(Tfig, βfig; layout = (2, 1), link = :x)
end

# ╔═╡ 963b5d26-1f93-4cd9-8a9b-989e22c16143
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 32955af4-fa47-43dd-b6a8-7545e1398f25
md"# Regret"

# ╔═╡ 39409585-662d-4915-ae42-653e35a2b975
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 9283bc5d-f5be-4b39-bc02-aeb942f1db79
#=╠═╡
plot(regretsol; idxs = 1, linewidth = 0.5, c = :black, opacity = 0.1)
  ╠═╡ =#

# ╔═╡ 68d51817-7f7e-4ebb-b986-67280479a999
#=╠═╡
plot(regretsol; idxs = 2, linewidth = 0.5, c = :black, opacity = 0.1)
  ╠═╡ =#

# ╔═╡ 498bf7bf-4812-4525-864e-db5454c53211
# ╠═╡ disabled = true
#=╠═╡
βregret = @closure (T, m, t) -> begin
	model = ifelse(T - hogg.Tᵖ > modelimminent.albedo.Tᶜ, modelimminent, modelremote)

	αitp = abatementmap[model]

	return β(t, ε(t, exp(m), αitp(T, m, t), model), model.economy)
end;
  ╠═╡ =#

# ╔═╡ f5584590-689c-4ddb-8e64-40c5c22e34ed
#=╠═╡
βM = Simulating.computeonsim(regretsol, βregret, yearlytime)
  ╠═╡ =#

# ╔═╡ 9c1452bc-611f-4e17-a9f5-89a80acf9568
#=╠═╡
let
	βM = Simulating.computeonsim(regretsol, βregret, yearlytime)
	βquantiles = Simulating.timequantiles(βM, [0.1, 0.5, 0.9])
    Simulating.smoothquantile!.(eachcol(βquantiles), 0)

	regretfig = plot(yearlytime, βquantiles[:, 2])
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─35f99656-f68a-455c-9042-b5afc5e7a4a8
# ╟─44eee19a-595a-42fe-9302-0f19df42388a
# ╠═b447bdc5-ea85-462c-a1f0-c8c8d5a3f1e9
# ╟─729c4427-986c-48b2-ae05-3b7b51c7d6bd
# ╟─6d43d072-4b34-11ef-3e45-8fde68196131
# ╠═d8c65298-2446-4135-b2a8-dbaf0aa90660
# ╠═5eb83b47-5f48-463d-8abb-21eaf36dbc25
# ╠═9ca8ccf3-c770-4d13-8ed5-ece9eaaf4b28
# ╠═cbd38547-eea1-4214-88c7-944c1aca82c2
# ╠═f6bab433-5d9d-4516-9c24-59c5441c06eb
# ╠═bfa6364c-7376-449e-90bc-11a37334092a
# ╠═542e01e0-bbce-462b-9c43-6d375adb6bd5
# ╟─a8afb34d-0273-4597-bbac-9650c172ed3b
# ╠═caba865f-055d-4655-91ca-14c537709dab
# ╠═e431c86c-c65e-48b0-ad78-50aeeb5ed63a
# ╟─cfad70a3-73d0-4d2f-9180-3f2c6322b17d
# ╠═9b215416-796e-4007-87bc-f6f0ed696b13
# ╟─80bfe18d-8d06-4957-a4ad-dd80bf8c42b1
# ╠═033b0310-2df1-4613-81f8-39274d2318ba
# ╟─e5895749-efc8-410a-bc1a-562979335a0b
# ╟─78a476ce-788c-428d-b6bf-da39f92f4035
# ╟─497f0b6b-e7ac-4176-81b3-f8df4050d338
# ╠═baa07798-cd01-49cb-8d33-eafd0c92505b
# ╠═3bbdb5f8-b258-4b93-b939-cad8d5c5fc11
# ╟─963b5d26-1f93-4cd9-8a9b-989e22c16143
# ╟─32955af4-fa47-43dd-b6a8-7545e1398f25
# ╠═39409585-662d-4915-ae42-653e35a2b975
# ╠═9283bc5d-f5be-4b39-bc02-aeb942f1db79
# ╠═68d51817-7f7e-4ebb-b986-67280479a999
# ╠═498bf7bf-4812-4525-864e-db5454c53211
# ╠═f5584590-689c-4ddb-8e64-40c5c22e34ed
# ╠═9c1452bc-611f-4e17-a9f5-89a80acf9568
