using Revise
using Test: @test
using BenchmarkTools: @btime
using Plots

includet("../terminal.jl")
includet("../utils/plotting.jl")

begin
	N = 21
	DATAPATH = "data"
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	hogg = Hogg()
	economy = Economy()
	albedo = Albedo()
	preferences = EpsteinZin()
	model = ModelInstance(preferences, economy, hogg, albedo, calibration);
end

domains = [
	(hogg.T₀, hogg.T₀ + 7), 
	(mstable(hogg.T₀, hogg, albedo), mstable(hogg.T₀ + 7., hogg, albedo)),
	(log(economy.Y₀ / 2), log(2economy.Y₀)), 
]

G = RegularGrid(domains, N);
V₀ = [log(exp(Xᵢ.y)) / preferences.ρ for Xᵢ ∈ G.X];
V₀ = (V₀ .- 2maximum(V₀)) / 2maximum(V₀); # Ensurate that V₀ < 0

begin
	V̄ = copy(V₀);
	policy = zeros(size(G));
	
	anim = @animate for iter in 1:120
		print("Plotting iteration $iter\r")
		terminaljacobi!(V̄, policy, model, G; indices = indices)
		sec = plotsection(V̄, log(economy.Y₀), G; zdim = 3, surf = true, c = :viridis, camera = (45, 45), yflip = false, xflip = true, title = "Iteration $iter")

	end

	gif(anim; fps = 12)
end