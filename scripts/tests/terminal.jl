using Revise
using Test: @test
using BenchmarkTools: @btime
using Plots

includet("../terminalproblem.jl")
includet("../utils/plotting.jl")

begin
	N = 31
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	hogg = Hogg()
	economy = Economy()
	albedo = Albedo(λ₂ = Albedo().λ₁)
	preferences = CRRA()
	model = ModelInstance(preferences, economy, hogg, albedo, calibration);
end

domains = [
	(hogg.T₀, hogg.T₀ + 5.), 
	(mstable(hogg.T₀, hogg, albedo), mstable(hogg.T₀ + 5., hogg, albedo)),
	(log(economy.Y₀ / 2), log(2economy.Y₀)), 
]

G = RegularGrid(domains, N);
V₀ = [log(exp(Xᵢ.y)) / preferences.ρ for Xᵢ ∈ G.X];

begin
	V̄ = copy(V₀); policy = zeros(size(G))
	indices = CartesianIndices(G)
	anim = @animate for iter in 1:120
		
		sec = plotsection(policy, log(economy.Y₀), G; zdim = 3, surf = true, c = :viridis, camera = (45, 45), yflip = false, xflip = true, title = "Iteration $iter")

		terminaljacobi!(V̄, policy, model, G; indices = isodd(iter) ? reverse(indices) : indices)
		print("Iteration $iter \r")
	end

	gif(anim; fps = 20)
end