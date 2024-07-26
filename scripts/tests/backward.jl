using Revise
using Test: @test
using BenchmarkTools: @btime
using JLD2
using Plots

includet("../utils/saving.jl")
includet("../terminal.jl")
includet("../backward.jl")

println("Startup with $(nprocs()) processes...")

begin
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	hogg = Hogg()
	economy = Economy()
	damages = LevelDamages()
	preferences = EpsteinZin()
	albedo = Albedo(λ₂ = Albedo().λ₁)
end

# --- Albedo
N = 51
model = TippingModel(albedo, preferences, damages, economy, hogg, calibration);
G = constructdefaultgrid(N, model);

# Testing the backward step
begin
	F̄, terminalpolicy = loadterminal(model, G);
	F = SharedMatrix(F̄);
	policy = SharedMatrix([Policy(χ, 0.) for χ ∈ terminalpolicy]);

	cluster = 1:N^2 .=> 0.
	i, δt = last(cluster)
	Δts = SharedVector(zeros(N^2))
	t = model.economy.τ

	Xᵢ = G.X[rand(CartesianIndices(G))]
	ᾱ = γ(t, model.calibration) + δₘ(exp(Xᵢ.m), model.hogg)

	u = Policy(rand(terminalpolicy), rand() * ᾱ)
end;

backwardstep!(Δts, F, policy, fullcluster, model, G)

F̄, terminalpolicy = loadterminal(model, G);
F = SharedMatrix(F̄);
policy = SharedMatrix([Policy(χ, 0.) for χ ∈ terminalpolicy]);
backwardsimulation!(F, policy, model, G; verbose = true)

# --- Jump
jump = Jump()
jumpmodel = JumpModel(jump, preferences, damages, economy, hogg, calibration);
F̄, terminalpolicy = loadterminal(model, G);

F = SharedMatrix(F̄);
policy = SharedMatrix([Policy(χ, 0.) for χ ∈ terminalpolicy]);
backwardsimulation!(F, policy, jumpmodel, G; verbose = true)

# -- Plot results
Tspace = range(G.domains[1]...; length = size(G, 1))
mspace = range(G.domains[2]...; length = size(G, 2))
begin
	consfig = wireframe(mspace, Tspace, first.(policy); camera = (45, 45), yflip = false, xflip = true, xlabel = "\$m\$", ylabel = "\$T\$", title = "\$\\chi\$")

	abatfig = wireframe(mspace, Tspace, last.(policy); camera = (45, 45), yflip = false, xflip = true, xlabel = "\$m\$", ylabel = "\$T\$", title = "\$\\alpha\$")

	plot(consfig, abatfig)
end

valuefig = wireframe(mspace, Tspace, F; camera = (45, 45), yflip = false, xflip = true, xlabel = "\$m\$", ylabel = "\$T\$", title = "\$F(T, m)\$")