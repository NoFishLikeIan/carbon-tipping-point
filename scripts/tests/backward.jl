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
	Δts = SharedVector(zeros(N^2))
	u = Policy(rand(terminalpolicy), rand())
end;

backwardstep!(Δts, F, policy, cluster, model, G)

F̄, terminalpolicy = loadterminal(model, G);
F = SharedMatrix(F̄);
policy = SharedMatrix([Policy(χ, 0.) for χ ∈ terminalpolicy]);
backwardsimulation!(F, policy, model, G; verbose = true)

computebackward(model, G; allownegative = true, verbose = true, datapath = "data", overwrite = true, tstop = 299.5, cachestep = 0.1)