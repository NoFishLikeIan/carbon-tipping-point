using Test, BenchmarkTools, Revise

using Model, Grid
using FastClosures
using ZigZagBoomerang
using Base.Threads
using SciMLBase
using Optim
using Statistics
using StaticArrays

using JLD2, UnPack
using Dates, Printf

includet("../utils/saving.jl")
includet("../markov/chain.jl")
includet("../markov/terminal.jl")

begin
	calibrationfilepath = "data/calibration.jld2"; @assert isfile(calibrationfilepath)

	calibrationfile = jldopen(calibrationfilepath, "r+")
	@unpack hogg, calibration, albedo = calibrationfile
	close(calibrationfile)
	
	damages = GrowthDamages()
	preferences = EpsteinZin()
	economy = Economy()
end;

model = TippingModel(albedo, hogg, preferences, damages, economy);
N = 100
G = terminalgrid(N, model)

F₀ = ones(size(G));
F̄ = copy(F₀);
terminalpolicy = similar(F̄);
errors = Inf .* ones(size(G));

terminaljacobi!(F̄, terminalpolicy, errors, model, G)
F̄, policy = vfi(F₀, model, G; maxiter = 10_000, verbose = 2)

# --- Jump
jump = Jump()
model = JumpModel(jump,  hogg, preferences, damages, economy);

F̄ = [(X.T / hogg.T₀)^2 + (X.m / log(hogg.M₀))^2 for X in G.X]
policy = zeros(size(G));

F̄, policy = vfi(F₀, model, G; maxiter = 10_000, verbose = true, alternate = true)