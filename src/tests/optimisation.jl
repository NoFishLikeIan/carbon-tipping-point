using Test
using Distributed, SharedArrays
addprocs(3);

@everywhere begin
    using UnPack
    using SharedArrays
    using LatinHypercubeSampling
end

@everywhere begin 
    include("../../src/utils/grids.jl")
    include("../../src/utils/derivatives.jl")
    include("../../src/model/pdes.jl")
end

# This is done on the side to avoid redefining constants
@everywhere include("../../src/model/init.jl")

# -- Generate state cube
statedomain = [
    (hogg.T₀, hogg.T̄, 21), 
    (log(hogg.M₀), log(hogg.M̄), 21), 
    (log(economy.Y̲), log(economy.Ȳ), 21)
];
Ω = makegrid(statedomain);
X = fromgridtoarray(Ω);

# -- Generate action square
m = 1200 # Number of points in policy grid
actiondomain = [(1f-3, 1f0 - 1f-3), (1f-3, 1f0 - 1f-3)];
P = makegrid(actiondomain, m);

# ---- Benchmarking
using BenchmarkTools

begin # Value function and derivatives
    V = rand(Float32, length.(Ω)) .- 1f0;
    ∇V = central∇(V, Ω);
    ∂²V = ∂²(V, Ω);
end;

# Some sample data
begin
    t = 5f0 
    idx = rand(CartesianIndices(V))
    jdx = rand(axes(P, 1))
    αᵢ, χᵢ = @view P[jdx, :]
    Xᵢ = @view X[idx, :]
    Vᵢ = @view V[idx]
    ∇Vᵢ = @view ∇V[idx, :]
    ∂²Vᵢ = @view ∂²V[idx]
end;

begin
    println("HJB given control...")
    @btime hjb($χᵢ, $αᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $∂²Vᵢ);

    println("Objective functional given control...")
    @btime objectivefunction($χᵢ, $αᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ);

    println("Optimal policy at a given point...")
    @btime optimalpolicy($t, $Xᵢ, $Vᵢ, $∇Vᵢ, $P);
end;

println("Optimal policy on the full grid...")
policysize = (size(V)..., 2);

println("Single core...")
nondistpolicy = Array{Float32, 4}(undef, policysize);
@btime policyovergrid!($nondistpolicy, $t, $X, $V, $∇V, $P);

println("$(nprocs()) cores...")
policy = SharedArray{Float32, 4}(policysize, init = S -> S .= NaN);
@btime policyovergrid!($policy, $t, $X, $V, $∇V, $P);

begin
    println("Asserting correctness...")

    c′ = optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ, P);
    @test all(nondistpolicy[idx, :] .== c′);
    @test all(policy[idx, :] .== c′);
    @test all(nondistpolicy .≈ policy)
end

# -- Benchmarking G
∂ₜV = similar(V);
w = Array{Float32}(undef, size(V)..., 3);
∇V = Array{Float32}(undef, size(V)..., 4);
@btime G($t, $X, $V, $Ω, $P);
@btime G!($∂ₜV, $∇V, $w, $policy, $t, $X, $V, $Ω, $P);

∂ₜV = vec(∂ₜV);
const V₀ = (-1f0 .- rand(Float32, length(V)));
function Ḡ!(∂ₜV::Vector{Float32}, V::Vector{Float32}, p, t)
    ∂ₜV .= vec(Ḡ(reshape(V, n, n, n), X, Ω, Γ)) .* (V .< 0)
end

@time Ḡ!(∂ₜV, V₀, [], 0f0);
prob = ODEProblem(Ḡ!, V₀, (0f0, 1f0));
sol = solve(prob)

terminalproblem = SteadyStateProblem(prob);