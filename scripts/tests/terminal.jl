using Revise
using Test: @test
using BenchmarkTools: @btime
using Plots

using Model, Grid

includet("../utils/saving.jl")
includet("../terminal.jl")

begin
	DATAPATH = "data"
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	hogg = Hogg()
	economy = Economy()
	preferences = EpsteinZin()
	albedo = Albedo()
end;

# --- Albedo
damages = GrowthDamages()
model = TippingModel(albedo, preferences, damages, economy, hogg, calibration);

begin
	N = 51
	Tupper = typeof(damages) <: LevelDamages ? hogg.T₀ + 8. : criticaltemperature(model)

	Tdomain = (hogg.Tᵖ, Tupper)
	mdomain = (mstable(Tdomain[1] - 0.75, hogg, albedo), mstable(Tdomain[2], hogg, albedo))

	domains = [Tdomain, mdomain]

	G = RegularGrid(domains, N);

	Tspace = range(G.domains[1]...; length = size(G, 1))
	mspace = range(G.domains[2]...; length = size(G, 2))
end

F₀ = ones(size(G)); F̄ = copy(F₀);
policy₀ = 0.5 * ones(size(G)); terminalpolicy = copy(policy₀);
terminaljacobi!(F̄, terminalpolicy, model, G)

F̄, policy = vfi(F₀, model, G; maxiter = 10_000, verbose = true)

begin
	idxs = 1:N
	mspacefig = mspace[idxs]
	Tspacefig = Tspace[idxs]

	Ffig = contourf(mspacefig, Tspacefig, log.(F̄[idxs, idxs]); xlabel = "\$m\$", ylabel = "\$T\$", linewidth = 0, dpi = 180)

	polfig = contourf(mspacefig, Tspacefig, policy[idxs, idxs]; xlabel = "\$m\$", ylabel = "\$T\$", linewidth = 0, dpi = 180)

	plot(Ffig, polfig; size = 600 .* (2√2, 1))
end

# --- Jump
jump = Jump()
model = JumpModel(jump, preferences, damages, economy, hogg, calibration);

F̄ = [(X.T / hogg.T₀)^2 + (X.m / log(hogg.M₀))^2 for X in G.X]
policy = zeros(size(G));

F̄, policy = vfi(F₀, model, G; maxiter = 10_000, verbose = true, alternate = true)

begin
	idxs = 1:N
	mspacefig = mspace[idxs]
	Tspacefig = Tspace[idxs]

	Ffig = contourf(mspacefig, Tspacefig, log.(F̄[idxs, idxs]); xlabel = "\$m\$", ylabel = "\$T\$", linewidth = 0, dpi = 180)

	polfig = contourf(mspacefig, Tspacefig, policy[idxs, idxs]; xlabel = "\$m\$", ylabel = "\$T\$", linewidth = 0, dpi = 180)

	plot(Ffig, polfig; size = 600 .* (2√2, 1))
end