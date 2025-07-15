using Test, BenchmarkTools, Revise, UnPack

using Model, Grid
using FastClosures
using ZigZagBoomerang
using Base.Threads
using SciMLBase
using Optim
using Statistics
using StaticArrays

using JLD2
using Dates, Printf

includet("../utils/saving.jl")
includet("../markov/terminal.jl")

begin
	calibration = load_object("data/calibration.jld2")
	hogg = Hogg()
	economy = Economy()
	preferences = EpsteinZin(ψ = 1.5, θ = 2.)
	albedo = Albedo(1.5)
end;

# --- Albedo
damages = GrowthDamages()
model = TippingModel(albedo, hogg, preferences, damages, economy);
N = 60
G = terminalgrid(N, model)

# F̄, terminalpolicy = loadterminal(model; outdir = "data/simulation/planner")

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