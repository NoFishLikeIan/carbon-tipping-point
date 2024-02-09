using Revise
using Test: @test
using BenchmarkTools: @btime
using JLD2

includet("../backward.jl")
include("../utils/plotting.jl")
include("../utils/saving.jl")

N, Δλ = 21, 0.
preferences = EpsteinZin()
name = filename(N, Δλ, preferences)
termpath = joinpath("data", "terminal", name)

termsim = load(termpath);
V̄ = termsim["V̄"];
terminalpolicy = termsim["policy"];
model = termsim["model"];
G = termsim["G"];

policy = SharedArray([Policy(χ, 0.) for χ ∈ terminalpolicy]);
V = SharedArray(deepcopy(V̄));

# cachepath = joinpath(DATAPATH, "test.jld2");
# @btime backwardsimulation!($V, $policy, $model, $G; t₀ = $model.economy.τ);