using Revise
using Test: @test
using BenchmarkTools: @btime

includet("../pde.jl")
include("../plotutils.jl")

N, Δλ = 51, 0.08
filename = "N=$(N)_Δλ=$(Δλ).jld2"
termpath = joinpath(DATAPATH, "terminal", filename)

termsim = load(termpath);
V̄ = SharedArray(termsim["V̄"]);
terminalpolicy = termsim["policy"];
model = termsim["model"];

policy = SharedArray([Policy(χ, 0.) for χ ∈ terminalpolicy]);
V = deepcopy(V̄);

cachepath = joinpath(DATAPATH, "test.jld2");

backwardsimulation!(V, policy, model; tmin = model.economy.τ, verbose = true);
@code_warntype backwardsimulation!(V, policy, model; tmin = model.economy.τ);
@btime backwardsimulation!($V, $policy, $model; tmin = model.economy.τ);