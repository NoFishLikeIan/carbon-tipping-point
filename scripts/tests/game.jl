using Revise
using Test: @test
using BenchmarkTools: @btime
using Plots

using Model, Grid
using JLD2

includet("../utils/saving.jl")
includet("../markov/terminal.jl")
includet("../markov/chain.jl")

begin
	DATAPATH = "data"
	rc = load_object(joinpath(DATAPATH, "regionalcalibration.jld2"))
	hogg = Hogg()
	economy = RegionalEconomies()
	preferences = (EpsteinZin(), EpsteinZin())
	albedo = Albedo(1.5)
	damages = (GrowthDamages(), GrowthDamages())
end;

model = TippingGameModel(albedo, hogg, preferences, damages, economy, rc);

begin
	N = 31
	G = terminalgrid(N, model)

	Tspace = range(G.domains[1]...; length = size(G, 1))
	mspace = range(G.domains[2]...; length = size(G, 2))
end

begin # Terminal problem
	hmodel, lmodel = breakgamemodel(model)

	computeterminal(hmodel, G; verbose = true, outdir = "data/game-test", addpath = "high")
	computeterminal(lmodel, G; verbose = true, outdir = "data/game-test", addpath = "low")
end