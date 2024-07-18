using Revise
using Test: @test
using BenchmarkTools: @btime
using JLD2
using Plots

includet("../backward.jl")
includet("../terminal.jl")
includet("../utils/saving.jl")

begin
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	hogg = Hogg()
	economy = Economy()
	damages = GrowthDamages()
	preferences = EpsteinZin()
end

begin
	N = 21

	albedo = Albedo();
	Tdomain = hogg.T₀ .+ (0., 7.)
	mdomain = (mstable(Tdomain[1] - 0.75, hogg, albedo), mstable(Tdomain[2], hogg, albedo))

	domains = [Tdomain, mdomain]

	G = RegularGrid(domains, N);
end

# --- Albedo
model = ModelInstance(preferences, economy, damages, hogg, albedo, calibration);
F̄, terminalpolicy = loadterminal(model, G);

F = SharedMatrix(F̄);
policy = SharedMatrix([Policy(χ, 0.) for χ ∈ terminalpolicy]);
backwardsimulation!(F, policy, model, G; verbose = true, t₀ = 0.)

# --- Jump
jumpmodel = ModelBenchmark(preferences, economy, damages, hogg, Jump(), calibration);
F̄, terminalpolicy = loadterminal(model, G);

F = SharedMatrix(F̄);
policy = SharedMatrix([Policy(χ, 0.) for χ ∈ terminalpolicy]);
backwardsimulation!(F, policy, jumpmodel, G; verbose = true, t₀ = 0.)

# -- Plot results
Tspace = range(G.domains[1]...; length = size(G, 1))
mspace = range(G.domains[2]...; length = size(G, 2))
begin
	consfig = wireframe(mspace, Tspace, first.(policy); camera = (45, 45), yflip = false, xflip = true, xlabel = "\$m\$", ylabel = "\$T\$", title = "\$\\chi\$")

	abatfig = wireframe(mspace, Tspace, last.(policy); camera = (45, 45), yflip = false, xflip = true, xlabel = "\$m\$", ylabel = "\$T\$", title = "\$\\alpha\$")

	plot(consfig, abatfig)
end

valuefig = wireframe(mspace, Tspace, F; camera = (45, 45), yflip = false, xflip = true, xlabel = "\$m\$", ylabel = "\$T\$", title = "\$F(T, m)\$")