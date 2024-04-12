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

# ╔═╡ b9cc9ff6-7fc8-11ee-3f84-d90577b8ac4a
# ╠═╡ show_logs = false
begin
    import Pkg
    Pkg.activate(Base.current_project())
    Pkg.instantiate()
end

# ╔═╡ ad627a95-169a-4c57-9ed4-d132fae5152f
using UnPack, JLD2, DotEnv

# ╔═╡ d6f96488-a12a-4461-9339-33314c49056a
using CSV, DataFrames

# ╔═╡ 70b25c8f-b704-4eb3-af84-16bdf7119c90
using Roots

# ╔═╡ 1471d3ee-6830-4955-9ba6-e5c004701794
using Model, Grid

# ╔═╡ aef78453-73c2-43b9-ad59-979574471c4d
using Plots, PlutoUI; default(size = 500 .* (√2, 1), dpi = 180, titlefontsize = 12, linewidth = 2)

# ╔═╡ 2a7f2675-34c0-4b4d-9d33-f46a97a5d3d7
using DifferentialEquations, DifferentialEquations.EnsembleAnalysis

# ╔═╡ cec96807-c5a6-43ff-acdb-025104edce93
using Interpolations

# ╔═╡ bdf4a3ab-d52a-4dc8-8c4a-3e009ec7e5cd
md"# Climate dynamics"

# ╔═╡ e76801ef-ef69-45cf-9477-d9c48c1b942f
md"## Imports"

# ╔═╡ 24c016a9-1628-46f4-9a6f-0a6c34de2d7e
TableOfContents()

# ╔═╡ 48b9fd38-0ac8-4efc-9e9f-8e83c9ea3199
begin
	const DATAPATH = "../data"
	const IPCCDATAPATH = joinpath(DATAPATH, "climate-data", "proj-median.csv")
	
	const economy = Economy()
	const calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	const ipccproj = CSV.read(IPCCDATAPATH, DataFrame)
end;

# ╔═╡ d0693c70-c6f1-4b78-b50f-0de2c4e096f5
begin
	const baseline = filter(:Scenario => isequal("SSP5 - Baseline"), ipccproj)
	const Mᵇ = linear_interpolation(calibration.years .- 2020, baseline[:, "CO2 concentration"]; extrapolation_bc = Line())
	const Tᵇ = linear_interpolation(calibration.years .- 2020, baseline[:, "Temperature"]; extrapolation_bc = Line())
end;

# ╔═╡ 544d662d-8a48-47cf-9fd3-bada252ee18d
md"
## Tipping point
"

# ╔═╡ 19280ece-4e94-4d21-958c-aab0c5229f14
function F!(du, u, p, t)
	hogg, albedo = p
	T, m = u

	du[1] = μ(T, m, hogg, albedo) / hogg.ϵ
	du[2] = γ(t, economy, calibration)
end;

# ╔═╡ 60592af4-38a3-446e-b08e-47967364aec3
function G!(Σ, u, p, t)
	hogg, albedo = p

	Σ[1] = hogg.σₜ / hogg.ϵ
	Σ[2] = 0.
end;

# ╔═╡ 8a38caf4-c79d-4dfb-aef6-c723495b483d
md"``\Delta\lambda:`` $(@bind Δλ Slider(0:0.01:0.1, default = 0., show_value = true))
"

# ╔═╡ d837589f-25a6-4500-b1ac-b20db496b485
begin
	albedo = Albedo(λ₂ = 0.31 - Δλ, T₂ = 291)
	hogg = calibrateHogg(albedo)

	Tspace = range(hogg.Tᵖ, hogg.Tᵖ + 13.; length = 101)
	nullcline = [Model.mstable(T, hogg, albedo) for T ∈ Tspace]
	
	u₀ = [ hogg.T₀, log(hogg.M₀) ]
	
	prob = SDEProblem(SDEFunction(F!, G!), u₀, (0., economy.t₁), (hogg, albedo))
end

# ╔═╡ 7e67776d-ac49-4a29-a601-d9418ce91e2e
solution = solve(EnsembleProblem(prob); trajectories = 6);

# ╔═╡ 1507326b-eab2-43c0-b981-ca0ca5aad997
begin
	timespan = range(0., economy.t₁; length = 101)
	path = timeseries_point_median(solution, timespan)
	
	simfig = plot(xlabel = "\$m\$", ylabel = "\$T\$", ylims = extrema(Tspace), xlims = (5.5, 7.5), yticks = (hogg.Tᵖ:(hogg.Tᵖ + 13.), 0:13))

	plot!(simfig, nullcline, Tspace; c = :black, linestyle = :dash, label = false)
	plot!(simfig, path[2, :], path[1, :]; c = :darkred, label = "Albedo")
	plot!(simfig, log.(Mᵇ.(timespan)), Tᵇ.(timespan) .+ hogg.Tᵖ; c = :black, linewidth = 3, label = "BaU")

	for simulation in solution
		path = simulation(timespan)

		plot!(simfig, path[2, :], path[1, :]; c = :darkred, label = false, alpha = 0.2)
	end

	simfig
	
end

# ╔═╡ 31c8af20-05e0-4355-937e-f0a0b3fc72d7
md"## Equivalent jump process"

# ╔═╡ b6795842-7e9f-44eb-8768-b6065f6cbda1
function jump(T)
	0.0568 * (T - hogg.Tᵖ) − 0.0577 - 0.0029 * (T - hogg.Tᵖ)^2
end;

# ╔═╡ 6ec0461e-a901-4bc6-8f0e-a4b1e7f89f67
function intensity(T)
	-0.25 + 0.95 / (1 + 2.8exp(-0.3325(T - hogg.Tᵖ)))
end;

# ╔═╡ 3b427901-b74b-4c64-a97e-f82711aa01cc
function affect!(integrator)
	integrator.u[1] += jump(integrator.u[1])
end;

# ╔═╡ c3ba9f6f-c240-43cb-a2a7-ef80423bfa1d
intensity(u, p, t) = intensity(u[1]);

# ╔═╡ 2ab85406-9736-43a2-b562-4e1c76f32feb
let
	plot(Tspace, jump; xlims = extrema(Tspace), ylabel = "Jump", yguidefontcolor = :darkred, c = :darkred, label = false, ylims = (0., 0.4), linewidth = 5);
	plot!(twinx(), Tspace, intensity; xlims = extrema(Tspace), ylabel = "Intensity", yguidefontcolor = :darkblue, c = :darkblue, label = false, ylims = (0, 0.7), linewidth = 5)
end

# ╔═╡ b1dff2b6-766a-4751-a4c5-ef2e84ec2bae
begin
	jumprate = VariableRateJump(intensity, affect!)
	jumpprob = JumpProblem(
		ODEProblem(F!, u₀, (0., economy.t₁), (hogg, Albedo(λ₂ = albedo.λ₁))),
		Direct(), jumprate
	)

	jumpsolution = solve(jumpprob, Tsit5())
end;

# ╔═╡ b2267725-c605-4100-bc31-29e04b8dc393
begin
	jumpfig = deepcopy(simfig)


	for simulation in jumpsolution
		path = simulation(timespan)

		plot!(jumpfig, path[2, :], path[1, :]; c = :darkblue, label = false, alpha = 0.2)
	end
 
	jumpfig
end

# ╔═╡ Cell order:
# ╟─bdf4a3ab-d52a-4dc8-8c4a-3e009ec7e5cd
# ╟─e76801ef-ef69-45cf-9477-d9c48c1b942f
# ╠═b9cc9ff6-7fc8-11ee-3f84-d90577b8ac4a
# ╠═ad627a95-169a-4c57-9ed4-d132fae5152f
# ╠═d6f96488-a12a-4461-9339-33314c49056a
# ╠═70b25c8f-b704-4eb3-af84-16bdf7119c90
# ╠═1471d3ee-6830-4955-9ba6-e5c004701794
# ╠═aef78453-73c2-43b9-ad59-979574471c4d
# ╠═24c016a9-1628-46f4-9a6f-0a6c34de2d7e
# ╠═2a7f2675-34c0-4b4d-9d33-f46a97a5d3d7
# ╠═cec96807-c5a6-43ff-acdb-025104edce93
# ╠═48b9fd38-0ac8-4efc-9e9f-8e83c9ea3199
# ╠═d0693c70-c6f1-4b78-b50f-0de2c4e096f5
# ╟─544d662d-8a48-47cf-9fd3-bada252ee18d
# ╠═19280ece-4e94-4d21-958c-aab0c5229f14
# ╠═60592af4-38a3-446e-b08e-47967364aec3
# ╟─8a38caf4-c79d-4dfb-aef6-c723495b483d
# ╠═d837589f-25a6-4500-b1ac-b20db496b485
# ╠═7e67776d-ac49-4a29-a601-d9418ce91e2e
# ╟─1507326b-eab2-43c0-b981-ca0ca5aad997
# ╟─31c8af20-05e0-4355-937e-f0a0b3fc72d7
# ╠═b6795842-7e9f-44eb-8768-b6065f6cbda1
# ╠═6ec0461e-a901-4bc6-8f0e-a4b1e7f89f67
# ╠═2ab85406-9736-43a2-b562-4e1c76f32feb
# ╠═3b427901-b74b-4c64-a97e-f82711aa01cc
# ╠═c3ba9f6f-c240-43cb-a2a7-ef80423bfa1d
# ╠═b1dff2b6-766a-4751-a4c5-ef2e84ec2bae
# ╠═b2267725-c605-4100-bc31-29e04b8dc393
