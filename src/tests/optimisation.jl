using Distributed, SharedArrays
addprocs(3);

@everywhere begin
    using UnPack 
    
    include("../../src/utils/grids.jl")
    include("../../src/utils/derivatives.jl")
    include("../../src/model/init.jl")
end

# -- Generate state cube
n = 51
const statedomain::Vector{Domain} = [
    (hogg.T₀, hogg.T̄, n), 
    (log(hogg.M₀), log(hogg.M̄), n), 
    (log(economy.Y̲), log(economy.Ȳ), n)
];
const Ω = makeregulargrid(statedomain);
const X = fromgridtoarray(Ω);

# -- Generate action square
m = 51
const actiondomain::Vector{Domain} = [
    (1f-3, 1f0 - 1f-3, m), (1f-3, 1f0 - 1f-3, m)
]

const Γ = makeregulargrid(actiondomain);
const P = fromgridtoarray(Γ);

# ---- Benchmarking
V = -rand(Float32, size(X[:, :, :, 1])) .- 1f0;
∇V = central∇(V, Ω);
∂²V = ∂²(1, V, Ω);

α = χ = 0.5f0;
objective = similar(V);
terminalpolicy = similar(V);
idx = rand(CartesianIndices(V));

t = 80f0;
Xᵢ = @view X[idx, :];
Vᵢ = @view V[idx];
∇Vᵢ = @view ∇V[idx, :];
∂²Vᵢ = @view ∂²V[idx];

using BenchmarkTools

begin
    println("HJB given control...")
    @btime hjb($χ, $α, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $∂²Vᵢ);

    println("Objective functional given control...")
    @btime objectivefunction($χ, $α, $t, $Xᵢ, $Vᵢ, $∇Vᵢ);

    println("Optimal policy at a given point...")
    @btime optimalpolicy($t, $Xᵢ, $Vᵢ, $∇Vᵢ, $Γ);
    @btime terminalpolicyovergrid!(terminalpolicy, $X, $V, $∇V, $Γ);


    # FIXME: the distributed value gives wrong values.
    println("Optimal policy on the full grid...")
    policy = SharedArray{Float32, 4}((size(V)..., 2));
    policyovergrid!(policy, t, X, V, ∇V, Γ); @time policyovergrid!(policy, t, X, V, ∇V, Γ);

    nondistpolicy = Array{Float32, 4}(undef, (size(V)..., 2));
    policyovergrid!(nondistpolicy, t, X, V, ∇V, Γ); @time policyovergrid!(nondistpolicy, t, X, V, ∇V, Γ);
    # @btime policyovergrid!(policy, $t, $X, $V, $∇V, $Γ); # FIXME: @btime not working for distributed
end;

if false
    ∂ₜV = similar(V);
    @btime Ḡ!(∂ₜV, $V, $X, $Ω, $Γ);

    ∂ₜV = vec(∂ₜV);
    const V₀ = (-1f0 .- rand(Float32, length(V)));
    function Ḡ!(∂ₜV::Vector{Float32}, V::Vector{Float32}, p, t)
        ∂ₜV .= vec(Ḡ(reshape(V, n, n, n), X, Ω, Γ)) .* (V .< 0)
    end

    @time Ḡ!(∂ₜV, V₀, [], 0f0);
    prob = ODEProblem(Ḡ!, V₀, (0f0, 1f0));
    sol = solve(prob)

    terminalproblem = SteadyStateProblem(prob);
end