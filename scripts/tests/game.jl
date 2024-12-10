using Revise
using Test: @test
using BenchmarkTools

using Model, Grid
using JLD2

includet("../utils/saving.jl")
includet("../markov/terminal.jl")
includet("../markov/game.jl")

begin
	DATAPATH = "data"

	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))

	hogg = Hogg()
	preferences = (EpsteinZin(), EpsteinZin())
	albedo = Albedo(1.5)
	damages = (GrowthDamages(), LevelDamages())
	
	oecdeconomy, roweconomy = RegionalEconomies()
	oecdmodel = TippingModel(albedo, hogg, EpsteinZin(), LevelDamages(), oecdeconomy)
	rowmodel = TippingModel(albedo, hogg, EpsteinZin(), GrowthDamages(), roweconomy)

	models = [oecdmodel, rowmodel]
end;

begin
	regionalcalibration = load_object(joinpath(DATAPATH, "regionalcalibration.jld2"))

	oecdcalibration = regionalcalibration[:oecd]
	rowcalibration = regionalcalibration[:row]
	regionalcalibrations = [oecdcalibration, rowcalibration]
end;


begin
	N = 31
	G = terminalgrid(N, oecdmodel)

	Tspace = range(G.domains[1]...; length = size(G, 1))
	mspace = range(G.domains[2]...; length = size(G, 2))
end

begin # Terminal problem
	computeterminal(oecdmodel, G; verbose = true, outdir = "data/game-test", addpath = "oecd", overwrite = true)
	computeterminal(rowmodel, G; verbose = true, outdir = "data/game-test", addpath = "row", overwrite = true)
end

# Backward step
terminalresults = loadterminal([oecdmodel, rowmodel]; outdir = "data/game-test", addpath = ["oecd", "row"]);

policies = Array{Float64, length(size(G)) + 1}[]
Fs = NTuple{2, Array{Float64, length(size(G))}}[]

for (i, terminalresult) in enumerate(terminalresults)
	F̄, terminalconsumption, terminalG = terminalresult
	Fₜ₊ₕ = interpolateovergrid(terminalG, G, F̄)
	Fₜ = similar(Fₜ₊ₕ)
	F = (Fₜ, Fₜ₊ₕ)

	policy = Array{Float64}(undef, size(G)..., 2)
	policy[:, :, 1] .= interpolateovergrid(terminalG, G, terminalconsumption)
	policy[:, :, 2] .= γ(models[i].economy.τ, regionalcalibrations[i])

	push!(policies, policy)
	push!(Fs, F)
end

begin
	queue = DiagonalRedBlackQueue(G)
	Δts = zeros(prod(size(G)))
	cluster = first(dequeue!(queue))
end;

backwardstep!(Δts, Fs, policies, cluster, model, regionalcalibrations, calibration, G) # FIXME: ArgumentError

# @benchmark backwardstep!($Δts, $Fs, $policies, $cluster, $model, $regionalcalibrations, $calibration, $G)