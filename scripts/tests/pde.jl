using Revise
using Test: @test
using BenchmarkTools: @btime

includet("../pde.jl")

V̄, terminalpolicy, model = getterminal(50, 0.08);

policy = [Policy(χ, 1e-5) for χ ∈ terminalpolicy];

V = copy(V̄);
backwardsimulation!(V, policy, model; tmin = 119.5, verbose = true);