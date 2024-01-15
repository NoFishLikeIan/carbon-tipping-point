using Revise
using Test: @test
using BenchmarkTools: @btime
using Plots

includet("../terminalproblem.jl")
includet("../plotutils.jl")

begin
	N = 50
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	hogg = Hogg()
	economy = Economy()
	albedo = Albedo()
end

# -- Critical domain
Tᶜ = find_zero(T -> bterminal(T, 0., economy, hogg), (hogg.Tᵖ, hogg.Tᵖ + 15.))

criticaldomains = [
	(Tᶜ, Tᶜ + 5.),
	(mstable(hogg.Tᵖ, hogg, albedo), mstable(Tᶜ + 5., hogg, albedo)),
	(log(economy.Y₀ * 0.01), log(3economy.Y₀)),
]

criticalgrid = RegularGrid(criticaldomains, N);
criticalindices = CartesianIndices(criticalgrid, Dict(3 => (true, false))); # Excludes the values y₀ in the iteration process

Vᶜ₀ = -ones(size(criticalgrid)); criticalpolicy = similar(Vᶜ₀);
criticalmodel = ModelInstance(economy, hogg, albedo, criticalgrid, calibration);

begin
	Vᶜ = copy(Vᶜ₀);
	anim = @animate for iter in 1:150
		sec = plotsection(Vᶜ, first(criticaldomains[2]), criticalmodel; zdim = 2, surf = true, c = :viridis, camera = (45, 45), yflip = true, xflip = false, title = "Iteration $iter")
		
		terminaljacobi!(Vᶜ, criticalpolicy, criticalmodel; indices = criticalindices)
		print("Iteration $iter \r")
	end

	gif(anim; fps = 15)
end

# -- Regular domain
domains = [
	(hogg.T₀, Tᶜ), 
	(mstable(hogg.T₀ - 0.5, hogg, albedo), mstable(Tᶜ + 0.5, hogg, albedo) + 1),
	(log(economy.Y₀ / 10), log(2economy.Y₀)), 
]

grid = RegularGrid(domains, N);
model = ModelInstance(economy, hogg, albedo, grid, calibration);

V₀ = min.(interpolateovergrid(criticalgrid, Vᶜ, grid), 0.)
policy = clamp.(interpolateovergrid(criticalgrid, criticalpolicy, grid), 0, 1)

begin
	lowerboundary = plotsection(Vᶜ, Tᶜ, criticalmodel; zdim = 1, c = :viridis, linewidth = 0, title = "\$V^{c}\$")


	sqr = Shape([
		(domains[3][1], domains[2][1]),
		(domains[3][2], domains[2][1]),
		(domains[3][2], domains[2][2]),
		(domains[3][1], domains[2][2])
	])
	plot!(lowerboundary, sqr, fillalpha = 0, linewidth = 3, c = :black, label = false)

	upperboundary = plotsection(V₀, Tᶜ, model; zdim = 1, c = :viridis, linewidth = 0, title = "\$V_0\$")

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
		sec = plotsection(V̄, log(hogg.M₀), model; zdim = 2, surf = true, c = :viridis, camera = (45, 45), yflip = false, xflip = true, title = "Iteration $iter")
		terminaljacobi!(V̄, policy, model; indices = indices)
		print("Iteration $iter \r")
	end

	gif(anim; fps = 15)
end