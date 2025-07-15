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
    calibration = load_object("data/calibration.jld2")
    damages = GrowthDamages()
    hogg = Hogg()
    preferences = EpsteinZin()
    economy = Economy()

    albedo = Albedo(1.5)

    model = TippingModel(albedo, hogg, preferences, damages, economy)
end

# Testing the backward step
F̄, terminalpolicy, G = try
	loadterminal(model; outdir = "data/simulation-local/constrained");
catch error
	@warn "Could not load terminal data, running terminal grid and VFI instead."

	N = 40
	G = terminalgrid(N, model)
	errors = Inf .* ones(size(G));
	F₀ = ones(size(G));

	F̄, terminalpolicy = vfi(F₀, model, G; maxiter = 10_000, verbose = 2)

	return F̄, terminalpolicy, G
end;

begin
	policy = Array{Float64}(undef, size(G)..., 2)
	policy[:, :, 1] .= terminalpolicy
	policy[:, :, 2] .= γ(economy.τ, calibration)
	Fₜ₊ₕ = copy(F̄);

	queue = DiagonalRedBlackQueue(G)
	Δts = zeros(prod(size(G)))
	cluster = first(ZigZagBoomerang.dequeue!(queue))

	Fₜ = similar(Fₜ₊ₕ)
	F = (Fₜ, Fₜ₊ₕ)
end;

withnegative = true
backwardstep!(Δts, F, policy, cluster, model, calibration, G; withnegative = withnegative);
@benchmark backwardstep!($Δts, $F, $policy, $cluster, $model, $calibration, $G; withnegative = $withnegative)
@profview backwardstep!(Δts, F, policy, cluster, model, calibration, G);
@profview_allocs backwardstep!(Δts, F, policy, cluster, model, calibration, G);