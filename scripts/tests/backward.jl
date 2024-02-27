using Revise
using Test: @test
using BenchmarkTools: @btime
using JLD2
using Plots

includet("../backward.jl")
includet("../utils/plotting.jl")
includet("../utils/saving.jl")

N = 21;
ΔΛ = [0., 0.08];
Θ = [3., 10.];
p = EpsteinZin(θ = last(Θ));

V̄, terminalpolicy, model, G = loadterminal(N, ΔΛ, p);
vspace = extrema(V̄)

policy = SharedArray([Policy(χ, 0.) for χ ∈ terminalpolicy[:, :, :, 1]]);
V = SharedArray(deepcopy(V̄[:, :, :, 1]));