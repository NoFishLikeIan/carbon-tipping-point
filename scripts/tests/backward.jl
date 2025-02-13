using Revise
using Test: @test
using UnPack: @unpack
using BenchmarkTools
using JLD2

includet("../utils/saving.jl")
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
begin
	F̄, terminalpolicy, G = loadterminal(model; outdir = "data/simulation-local/constrained");
	policy = Array{Float64}(undef, size(G)..., 2)
	policy[:, :, 1] .= terminalpolicy
	policy[:, :, 2] .= γ(economy.τ, calibration)

	Fₜ₊ₕ = copy(F̄);
end;

begin
	queue = DiagonalRedBlackQueue(G)
	Δts = zeros(prod(size(G)))
	cluster = first(dequeue!(queue))

	Fₜ = similar(Fₜ₊ₕ)
	F = (Fₜ, Fₜ₊ₕ)
end;

backwardstep!(Δts, F, policy, cluster, model, calibration, G);
@btime backwardstep!($Δts, $F, $policy, $cluster, $model, $calibration, $G);
@profview backwardstep!(Δts, F, policy, cluster, model, calibration, G);
@profview_allocs backwardstep!(Δts, F, policy, cluster, model, calibration, G);