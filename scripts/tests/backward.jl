using Revise
using Test: @test
using BenchmarkTools: @btime
using JLD2
using Plots

includet("../utils/saving.jl")
includet("../markov/terminal.jl")
includet("../markov/backward.jl")

println("Startup with $(nprocs()) processes...")

begin
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	hogg = Hogg()
	economy = Economy()
	damages = GrowthDamages()
	preferences = EpsteinZin()
	albedo = Albedo()
end

# --- Albedo
N = 51
model = TippingModel(albedo, hogg, preferences, damages, economy, calibration);
G = constructdefaultgrid(N, model);

# Testing the backward step
begin
	F̄, terminalpolicy = loadterminal(model);
	F = SharedMatrix(F̄);
	policy = SharedMatrix([Policy(χ, 0.) for χ ∈ terminalpolicy]);

	cluster = 1:N^2 .=> 0.
	Δts = SharedVector(zeros(N^2))
	i, δt = first(cluster)
end;

backwardstep!(Δts, F, policy, cluster, model, G)

F̄, terminalpolicy = loadterminal(model);
F = SharedMatrix(F̄);
policy = SharedMatrix([Policy(χ, 0.) for χ ∈ terminalpolicy]);

backwardstep!(Δts, F, policy, cluster, model, G; s = 1.)
heatmap(last.(policy), clims = (0, Inf), c = :Greens)

# backwardsimulation!(F, policy, model, G; verbose = true, s = 1)

# computebackward(model, G; verbose = true, tstop = economy.τ - 0.1)