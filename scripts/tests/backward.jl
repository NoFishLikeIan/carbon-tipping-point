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
    preferences = EpsteinZin(θ = 2., ψ = 0.75)
    economy = Economy()
    albedo = Albedo(1.5)

    model = TippingModel(albedo, hogg, preferences, damages, economy, calibration)
end

begin
	N = 101
	Tdomain = hogg.Tᵖ .+ (0., 9.);
	mdomain = mstable.(Tdomain, hogg)
	G = RegularGrid([Tdomain, mdomain], N)
end;

# Testing the backward step
begin
	F̄, terminalpolicy, Gterm = loadterminal(model; outdir = "data/test-simulation/constrained");
	policy = Array{Float64}(undef, size(G)..., 2)
	policy[:, :, 1] .= interpolateovergrid(Gterm, G, terminalpolicy)
	policy[:, :, 2] .= γ(economy.τ, calibration)

	F = interpolateovergrid(Gterm, G, F̄);
end;

begin
	queue = DiagonalRedBlackQueue(G)
	Δts = zeros(N^2)
	cluster = first(dequeue!(queue))
end;

begin
	b = BenchmarkTools.@benchmark backwardstep!($Δts, $F, $policy, $cluster, $model, $G)

	io = IOBuffer()
	show(io, "text/plain", b)
	s = String(take!(io))

	println(s)
end