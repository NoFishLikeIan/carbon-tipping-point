using Revise
using Test: @test
using BenchmarkTools: @btime
using JLD2
using Plots

includet("../utils/saving.jl")
includet("../markov/terminal.jl")
includet("../markov/game.jl")

begin
	rc = load_object(joinpath(DATAPATH, "regionalcalibration.jld2"))
	hogg = Hogg()
	economy = Economy()
	damages = GrowthDamages()
	preferences = EpsteinZin()
	albedo = Albedo()
end

# --- Albedo
N = 51
model = TippingGameModel(albedo, preferences, damages, economy, hogg, rc);
G = constructdefaultgrid(N, model);

# Testing the backward step
begin
	F̄, terminalpolicy = loadterminal(model, G);

	sizemat = (size(G, 1), size(G, 2), N ÷ 4, N ÷ 4)

	Fₕ = SharedArray{Float64}(sizemat);
	polₕ = SharedArray{Policy}(sizemat);

	for i ∈ 1:sizemat[3], j ∈ 1:sizemat[4]
		Fₕ[:, :, i, j] .= F̄
		polₕ[:, :, i, j] .= [Policy(χ, 0.) for χ ∈ terminalpolicy]
	end

	cluster = 1:N^2 .=> 0.
	Δts = SharedVector(zeros(N^2))
	i, δt = first(cluster)
end;

backwardstep!(Δts, F, policy, cluster, model, G)

F̄, terminalpolicy = loadterminal(model, G);
F = SharedMatrix(F̄);
policy = SharedMatrix([Policy(χ, 0.) for χ ∈ terminalpolicy]);

backwardstep!(Δts, F, policy, cluster, model, G; s = 1.)
heatmap(last.(policy), clims = (0, Inf), c = :Greens)

# backwardsimulation!(F, policy, model, G; verbose = true, s = 1)

# computebackward(model, G; verbose = true, tstop = economy.τ - 0.1)