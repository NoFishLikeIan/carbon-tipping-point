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

	Tᶜ = 1.5
    albedo = Albedo(Tᶜ)

    model = TippingModel(albedo, hogg, preferences, damages, economy, calibration)
end

begin
	N = 31
	Tdomain = hogg.Tᵖ .+ (0., 7.);
	mdomain = mstable.(Tdomain, hogg)
	G = RegularGrid([Tdomain, mdomain], N)
end;

# Testing the backward step
begin
	F̄, terminalpolicy, Gterm = loadterminal(model; outdir = "data/simulation-medium/constrained");
	policy = Array{Float64}(undef, size(G)..., 2)
	policy[:, :, 1] .= interpolateovergrid(Gterm, G, terminalpolicy)
	policy[:, :, 2] .= γ(economy.τ, calibration)

	Fₜ₊ₕ = interpolateovergrid(Gterm, G, F̄);
end;

begin
	queue = DiagonalRedBlackQueue(G)
	Δts = zeros(N^2)
	cluster = first(dequeue!(queue))

	Fₜ = similar(Fₜ₊ₕ)
	F = (Fₜ, Fₜ₊ₕ)
end;

backwardstep!(Δts, F, policy, cluster, model, G);

@btime backwardstep!($Δts, $F, $policy, $cluster, $model, $G; allownegative = true);
@btime backwardstep!($Δts, $F, $policy, $cluster, $model, $G; allownegative = false);
