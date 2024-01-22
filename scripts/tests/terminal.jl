using Revise
using Test: @test
using BenchmarkTools: @btime
using Plots

includet("../terminalproblem.jl")
includet("../plotutils.jl")

begin
	N = 11
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	hogg = Hogg()
	economy = Economy()
	albedo = Albedo()
	preferences = CRRA()
	model = ModelInstance(preferences, economy, hogg, albedo, calibration);
end

# -- Critical domain
Tᶜ = find_zero(T -> bterminal(T, 0., economy, hogg), (hogg.Tᵖ, hogg.Tᵖ + 15.))

criticaldomains = [
	(Tᶜ, Tᶜ + 5.),
	(mstable(hogg.Tᵖ, hogg, albedo), mstable(Tᶜ + 5., hogg, albedo)),
	(log(economy.Y₀ / 2), log(3economy.Y₀)),
]

criticalgrid = RegularGrid(criticaldomains, N);
criticalindices = CartesianIndices(criticalgrid, Dict(3 => (true, false))); # Excludes the values y₀ in the iteration process

Vᶜ₀ = -100ones(size(criticalgrid)); 
criticalpolicy = similar(Vᶜ₀);

begin
	Vᶜ = copy(Vᶜ₀);
	anim = @animate for iter in 1:150
		sec = plotsection(Vᶜ, log(hogg.M₀), criticalgrid; zdim = 2, surf = true, c = :viridis, camera = (45, 45), yflip = false, xflip = true, title = "Iteration $iter")
		
		terminaljacobi!(Vᶜ, criticalpolicy, model, criticalgrid; indices = criticalindices)
		print("Iteration $iter \r")
	end

	gif(anim; fps = 15)
end

# -- Regular domain
domains = [
	(hogg.T₀, Tᶜ), 
	(mstable(hogg.T₀ - 0.5, hogg, albedo), mstable(Tᶜ + 0.5, hogg, albedo) + 1),
	(log(economy.Y₀ / 2), log(2economy.Y₀)), 
]

standardgrid = RegularGrid(domains, N);

V₀ = min.(interpolateovergrid(criticalgrid, Vᶜ, standardgrid), 0.);
policy = interpolateovergrid(criticalgrid, criticalpolicy, standardgrid);

begin
	lowerboundary = plotsection(Vᶜ, Tᶜ, criticalgrid; zdim = 1, c = :viridis, linewidth = 0, title = "\$V^{c}\$")


	sqr = Shape([
		(domains[3][1], domains[2][1]),
		(domains[3][2], domains[2][1]),
		(domains[3][2], domains[2][2]),
		(domains[3][1], domains[2][2])
	])
	plot!(lowerboundary, sqr, fillalpha = 0, linewidth = 3, c = :black, label = false)

	upperboundary = plotsection(V₀, Tᶜ, standardgrid; zdim = 1, c = :viridis, linewidth = 0, title = "\$V_0\$")

	lowerboundary
end

# Excludes the values Tᶜ in the iteration process
indices = permutedims(
	reverse(CartesianIndices(criticalgrid, Dict(1 => (false, true)))),
	(3, 2, 1)
)

begin
	V̄ = copy(V₀);
	anim = @animate for iter in 1:45
		sec = plotsection(V̄, log(hogg.M₀) + 1., standardgrid; zdim = 2, surf = true, c = :viridis, camera = (45, 45), yflip = false, xflip = true, title = "Iteration $iter")
		terminaljacobi!(V̄, policy, model, standardgrid; indices = indices)
		print("Iteration $iter \r")
	end

	gif(anim; fps = 15)
end