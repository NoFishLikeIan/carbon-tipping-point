using Revise
using Test: @test
using BenchmarkTools
using JLD2
using Plots

using Distributed: nprocs

includet("../utils/saving.jl")
includet("../markov/terminal.jl")
includet("../markov/backward.jl")

println("Startup with $(nprocs()) processes...")

begin
	env = DotEnv.config()
	DATAPATH = get(env, "DATAPATH", "data")
	SIMULATIONPATH = get(env, "SIMULATIONPATH", "sim")

	datapath = joinpath(DATAPATH, SIMULATIONPATH)
end

begin
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	hogg = Hogg()
	economy = Economy()
	damages = GrowthDamages()
	preferences = EpsteinZin()
	albedo = Albedo(1.5)
end

begin
	model = TippingModel(albedo, hogg, preferences, damages, economy, calibration)

	N = 21
	Tdomain = hogg.Tᵖ .+ (0., 7.);
	mdomain = (mstable(Tdomain[1], hogg), mstable(Tdomain[2], hogg))
	G = RegularGrid([Tdomain, mdomain], N)
end;

# Testing the backward step
begin
	F̄, terminalpolicy, Gterm = loadterminal(model; datapath);
	policy = SharedArray{Float64}(size(G)..., 2)
	policy[:, :, 1] .= interpolateovergrid(Gterm, G, terminalpolicy)
	policy[:, :, 2] .= γ(economy.τ, calibration)

	F = interpolateovergrid(Gterm, G, F̄) |> SharedMatrix
end; 

begin
	queue = DiagonalRedBlackQueue(G)
	Δts = zeros(N^2) |> SharedVector
	cluster = first(dequeue!(queue))

	@sync backwardstep!(Δts, F, policy, cluster, model, G)
end;

begin
	b = @benchmark backwardstep!($Δts, $F, $policy, $cluster, $model, $G)
	io = IOBuffer()
	show(io, "text/plain", b)
	s = String(take!(io))
	println(s)
end