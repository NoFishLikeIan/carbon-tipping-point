using UnPack
using StatsBase
using LinearAlgebra

using BenchmarkTools

include("../../src/utils/grids.jl")
include("../../src/utils/derivatives.jl")
include("../../src/model/init.jl")

# -- Generate state cube
statedomain::Vector{Domain} = [
    (hogg.T₀, hogg.T̄, 101), 
    (log(hogg.M₀), log(hogg.M̄), 61), 
    (log(economy.Y̲), log(economy.Ȳ), 61)
];
Ω = makeregulargrid(statedomain);
Nₛ = size(Ω);
X = fromgridtoarray(Ω);

# -- Generate action square
m = 21
actiondomain::Vector{Domain} = [
    (1f-3, 1f0 - 1f-3, 11), (1f-3, 1f0 - 1f-3, 11)
]

Γ = makeregulargrid(actiondomain);
Nₐ = size(Γ);
P = fromgridtoarray(Γ);

V = -rand(Float32, size(X[:, :, :, 1])) .- 1f0;
∇V = central∇(V, Ω);

t = 0f0;
α = χ = 0.5f0;
objective = similar(V);
idx = rand(CartesianIndices(V));

Xᵢ = @view X[idx, :]
∇Vᵢ = @view ∇V[idx, :]
Vᵢ = @view V[idx]
t = 80f0


@btime control($χ, $α, $t, $Xᵢ, $Vᵢ, $∇Vᵢ);
@btime objectivefunctional!(objective, $χ, $α, $t, $X, $V, $∇V);