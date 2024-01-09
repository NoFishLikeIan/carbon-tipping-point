using Revise
using Test: @test
using BenchmarkTools: @btime

includet("../pde.jl")

N, Δλ = 50, 0.08
filename = "N=$(N)_Δλ=$(Δλ).jld2"
termpath = joinpath(DATAPATH, "terminal", filename)

termsim = load(termpath)
V̄ = termsim["V̄"]
terminalpolicy = termsim["policy"]
model = termsim["model"]

policy = [Policy(χ, 1e-5) for χ ∈ terminalpolicy]
V = copy(V̄)

cachepath = "test.jld2"

backwardsimulation!(V, policy, model; tmin = 119.5, verbose = true, cachepath = cachepath);