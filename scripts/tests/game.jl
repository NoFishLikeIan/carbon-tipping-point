using Revise
using Test: @test
using BenchmarkTools: @btime
using Plots

using Model, Grid

includet("../utils/saving.jl")
includet("../markov/terminal.jl")
includet("../markov/game.jl")

begin
	DATAPATH = "data"
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	rc = load_object(joinpath(DATAPATH, "regionalcalibration.jld2"))
	hogg = Hogg()
	economy = RegionalEconomies()
	preferences = (EpsteinZin(), EpsteinZin())
	albedo = Albedo(Tᶜ = 1.5)
	damages = (GrowthDamages(), GrowthDamages())
end;

# --- Albedo
plannermodel = TippingModel(albedo, hogg, first(preferences), first(damages), first(economy), calibration)
model = TippingGameModel(albedo, hogg, preferences, damages, economy, rc);

begin
	N = 31
	M = 10
	G = constructdefaultgrid(N, model)

	Tspace = range(G.domains[1]...; length = size(G, 1))
	mspace = range(G.domains[2]...; length = size(G, 2))
end

begin
	F̄, terminalpolicy = loadterminal(collect(breakgamemodel(model)), G; datapath = "data/simulation/game", addpath = ["high", "low"])

	N₁, N₂ = size(G)
	nmodels = size(F̄, 3)

	Fs = ntuple(_ -> SharedArray{Float64}(N₁, N₂, M), nmodels)
	policies = ntuple(_ -> SharedArray{Policy}(N₁, N₂, M), nmodels)

	for m in eachindex(Fs), k in axes(Fs[m], 3)
		Fs[m][:, :, k] .= F̄[:, :, m]
		policies[m][:, :, k] .= [Policy(χ, 0.) for χ ∈ terminalpolicy[:, :, m]]
	end
end