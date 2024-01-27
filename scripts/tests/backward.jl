using Revise
using Test: @test
using BenchmarkTools: @btime

includet("../backward.jl")
include("../utils/plotting.jl")

N, Δλ = 21, 0.
name = filename(N, Δλ, LogUtility())
termpath = joinpath(DATAPATH, "terminal", name)

termsim = load(termpath);
V̄ = SharedArray(termsim["V̄"]);
terminalpolicy = termsim["policy"];
model = termsim["model"];
G = termsim["G"];

policy = SharedArray([Policy(χ, 0.) for χ ∈ terminalpolicy]);
V = deepcopy(V̄);

cachepath = joinpath(DATAPATH, "test.jld2");

@btime backwardsimulation!($V, $policy, $model, $G; t₀ = $model.economy.τ);