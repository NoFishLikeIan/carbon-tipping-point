using Revise
using Test: @test
using BenchmarkTools: @btime
using Plots

using Model, Grid

includet("../terminal.jl")

begin
	DATAPATH = "data"
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	hogg = Hogg()
	economy = Economy()
	preferences = EpsteinZin(θ = 2.)

	leveldamages = LevelDamages()
	growthdamages = GrowthDamages()
	albedo = Albedo()
	jump = Jump()
end

begin
	N = 51
	Tdomain = hogg.T₀ .+ (0., 3.)
	mdomain = (mstable(Tdomain[1] - 0.75, hogg, albedo), mstable(Tdomain[2], hogg, albedo))

	domains = [Tdomain, mdomain]

	G = RegularGrid(domains, N);
end

# --- Albedo
begin
	modellevel = TippingModel(albedo, preferences, leveldamages, economy, hogg, calibration);
	modelgrowth = TippingModel(albedo, preferences, growthdamages, economy, hogg, calibration);
end

# Tests
K₀ = ones(size(G, 1)); K = copy(K₀);
policy = similar(K₀);

steadystatestep!(K, policy, modellevel, G)

Klevel = copy(K₀); Kgrowth = copy(K₀);
policylevel = copy(K₀); policygrowth = copy(K₀);
Tspace = range(G.domains[1]...; length = size(G, 1))

begin
	anim = @animate for iter in 1:120
		print("Plotting iteration $iter\r")

		steadystatestep!(Klevel, policylevel, modellevel, G)
		levelfig = plot(Tspace, Klevel; camera = (45, 45), title = "Level", xlabel = "\$T\$", ylims = (0, 2))
		
		steadystatestep!(Kgrowth, policygrowth, modelgrowth, G)
		growthfig = plot(Tspace, Kgrowth; camera = (45, 45), title = "Growth, iteration $iter", xlabel = "\$T\$", ylims = (0, 2))

		plot(levelfig, growthfig)
	end

	gif(anim; fps = 15)
end

K, policy = vfi(K₀, modelgrowth, G; verbose = true, alternate = true, tol = 1e-5, maxiter = 100_000)