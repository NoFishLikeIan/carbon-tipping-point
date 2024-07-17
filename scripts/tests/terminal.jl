using Revise
using Test: @test
using BenchmarkTools: @btime
using Plots

using Model, Grid

includet("../terminal.jl")
includet("../utils/plotting.jl")

begin
	DATAPATH = "data"
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	hogg = Hogg(σₜ = 0.5, σₘ = 0.5)
	economy = Economy()
	damages = GrowthDamages()
	preferences = EpsteinZin()
end

begin
	N = 41

	albedo = Albedo();
	Tdomain = hogg.T₀ .+ (0., 9.)
	mdomain = (mstable(Tdomain[1] - 0.75, hogg, albedo), mstable(Tdomain[2], hogg, albedo))

	domains = [Tdomain, mdomain]

	G = RegularGrid(domains, N);
end

F₀ = ones(size(G)); F̄ = copy(F₀);

# --- Albedo
model = ModelInstance(preferences, economy, damages, hogg, albedo, calibration);

F̄ = copy(F₀);
policy = zeros(size(G));
Tspace = range(G.domains[1]...; length = size(G, 1))
mspace = range(G.domains[2]...; length = size(G, 2))

begin

	anim = @animate for iter in 1:240
		print("Plotting iteration $iter\r")

		terminaljacobi!(F̄, policy, model, G)
		wireframe(mspace, Tspace, F̄; camera = (45, 45), yflip = false, xflip = true, title = "Iteration $iter", xlabel = "\$m\$", ylabel = "\$T\$")
	end

	gif(anim; fps = 60)
end

# --- Jump
jumpmodel = ModelBenchmark(preferences, economy, damages, hogg, Jump(), calibration);


F̄ = [(X.T / hogg.T₀)^2 + (X.m / log(hogg.M₀))^2 for X in G.X]
policy = zeros(size(G));

begin
	anim = @animate for iter in 1:60
		print("Plotting iteration $iter\r")

		terminaljacobi!(F̄, policy, jumpmodel, G)
		wireframe(mspace, Tspace, F̄; camera = (45, 45), yflip = false, xflip = true, title = "Iteration $iter", xlabel = "\$m\$", ylabel = "\$T\$")
	end

	gif(anim; fps = 12)
end

F̄, policy = vfi(F₀, jumpmodel, G; verbose = true, tol = 1e-4, alternate = true, maxiter = 1_000)

begin
	Ffig = wireframe(mspace, Tspace, F̄; camera = (45, 45), yflip = false, xflip = true, xlabel = "\$m\$", ylabel = "\$T\$")

	polfig = wireframe(mspace, Tspace, policy; camera = (45, 45), yflip = false, xflip = true, xlabel = "\$m\$", ylabel = "\$T\$")

	plot(Ffig, polfig)
end