using Test, BenchmarkTools, Revise, UnPack

using JLD2
using Printf

using Model, Grid
using FastClosures
using ZigZagBoomerang
using Base.Threads
using SciMLBase
using Optim
using Statistics
using StaticArrays

using Dates

includet("../utils/saving.jl")
include("../markov/chain.jl")
includet("../markov/terminal.jl")
includet("../markov/backward.jl")

begin
	calibrationfilepath = "data/calibration.jld2"; @assert isfile(calibrationfilepath)

	calibrationfile = jldopen(calibrationfilepath, "r+")
	@unpack hogg, calibration, albedo = calibrationfile
	close(calibrationfile)
	
	damages = GrowthDamages()
	preferences = EpsteinZin()
	economy = Economy()

	model = TippingModel(albedo, hogg, preferences, damages, economy)
end;

# Testing the backward step
F̄, terminalpolicy, G = try
	loadterminal(model; outdir = "data/simulation-local/constrained");
catch error
	@warn "Could not load terminal data, running terminal grid and VFI instead."

	N = 100
	G = terminalgrid(N, model)
	errors = Inf .* ones(size(G));
	F₀ = ones(size(G));

	F̄, terminalpolicy = vfi(F₀, model, G; maxiter = 10_000, verbose = 2)

	return F̄, terminalpolicy, G
end;

begin
	policy = [MVector{2}(terminalpolicy[idx], γ(economy.τ, calibration)) for idx in CartesianIndices(G)]

	queue = DiagonalRedBlackQueue(G)
	Δts = zeros(prod(size(G)))
	cluster = first(ZigZagBoomerang.dequeue!(queue))

	Fₜ = similar(Fₜ₊ₕ)
	F = (Fₜ, Fₜ₊ₕ)
end;

withnegative = true
backwardstep!(Δts, F, policy, cluster, model, calibration, G; withnegative = false)
@benchmark backwardstep!($Δts, $F, $policy, $cluster, $model, $calibration, $G; withnegative = $(true))