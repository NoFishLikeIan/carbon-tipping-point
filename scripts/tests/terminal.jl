using Revise
using Test: @test
using BenchmarkTools: @btime
using Plots

includet("../terminal.jl")
includet("../utils/plotting.jl")

begin
	DATAPATH = "data"
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	hogg = Hogg()
	economy = Economy()
	preferences = EpsteinZin()
end

begin
	N = 21

	Tdomain = hogg.T₀ .+ (0., 7.)
	mdomain = mstable.(Tdomain, Ref(hogg), Ref(Albedo()))
	ydomain = log.(economy.Y₀ .* (0.5, 2.))
	
	domains = [Tdomain, mdomain, ydomain]

	G = RegularGrid(domains, N);
end

# --- Albedo
albedo = Albedo();
model = ModelInstance(preferences, economy, hogg, albedo, calibration);

V₀ = [log(exp(Xᵢ.y)) / preferences.ρ for Xᵢ ∈ G.X];
V₀ = (V₀ .- 2maximum(V₀)) / 2maximum(V₀); # Ensurate that V₀ < 0

begin
	V̄ = copy(V₀);
	policy = zeros(size(G));
	
	anim = @animate for iter in 1:120
		print("Plotting iteration $iter\r")
		terminaljacobi!(V̄, policy, model, G)
		sec = plotsection(policy, log(economy.Y₀), G; zdim = 3, surf = true, c = :viridis, camera = (45, 45), yflip = false, xflip = true, title = "Iteration $iter")

	end

	gif(anim; fps = 12)
end

# --- Jump
jump = Jump()
modeljump = ModelBenchmark(preferences, economy, hogg, jump, calibration);

begin
	V̄ = copy(V₀);
	policy = zeros(size(G));
	
	anim = @animate for iter in 1:120
		print("Plotting iteration $iter\r")
		terminaljacobi!(V̄, policy, modeljump, G)
		sec = plotsection(policy, log(economy.Y₀), G; zdim = 3, surf = true, c = :viridis, camera = (45, 45), yflip = false, xflip = true, title = "Iteration $iter")

	end

	gif(anim; fps = 12)
end