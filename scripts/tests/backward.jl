using Revise
using Test: @test
using BenchmarkTools: @btime

includet("../backward.jl")
include("../plotutils.jl")

N, Δλ = 11, 0.08
filename = "N=$(N)_Δλ=$(Δλ).jld2"
termpath = joinpath(DATAPATH, "terminal", filename)

termsim = load(termpath);
V̄ = SharedArray(termsim["V̄"]);
terminalpolicy = termsim["policy"];
model = termsim["model"];
G = termsim["grid"];

policy = SharedArray([Policy(χ, 0.) for χ ∈ terminalpolicy]);
V = deepcopy(V̄);

cachepath = joinpath(DATAPATH, "test.jld2");

@btime backwardsimulation!($V, $policy, $model, $G; t₀ = $model.economy.τ);