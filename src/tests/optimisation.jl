using Test

using UnPack
using Polyester
using Optim

begin 
    include("../utils/grids.jl")
    include("../utils/derivatives.jl")
    include("../model/pdes.jl")
end

# This is done on the side to avoid redefining constants
include("../model/initialisation.jl")

# -- Generate state cube
statedomain = [
    (hogg.T₀, hogg.T̄, 101), 
    (log(hogg.M₀), log(hogg.M̄), 51), 
    (log(economy.Y̲), log(economy.Ȳ), 51)
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
    function vguess(Xᵢ)
        ((exp(Xᵢ[3]) / economy.Ȳ)^2 - (exp(Xᵢ[2]) / hogg.Mᵖ)^2 *  (Xᵢ[1] / hogg.Tᵖ)^2)
    end

    V = Array{Float32}(undef, length.(Ω));
    for I ∈ CartesianIndices(V) V[I] = vguess(X[I, :]) end
    V .= 1f-1 .* (V ./ maximum(abs.(V)))

    ∇V = central∇(V, Ω);
    ∂²V = ∂²(V, Ω);
end;

# Some sample data
begin
    t = 5f0
    idx = rand(CartesianIndices(V))
    Xᵢ = @view X[idx, :]
    Vᵢ = @view V[idx]
    ∇Vᵢ = @view ∇V[idx, :]
    ∂²Vᵢ = @view ∂²V[idx]

    jdx = rand(axes(P, 1))
    cᵢ = @view P[jdx, :]
end;

begin
    unit = range(1f-2, 1f0 - 1f-2; length = 101);
    g = objective(t, Xᵢ, Vᵢ, ∇Vᵢ);
    surface(unit, unit, (χ, α) -> -g([χ, α]); xlabel = "\$\\chi\$", ylabel = "\$\\alpha\$")
end

begin
    println("HJB given control...")
    @btime hjb($cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $∂²Vᵢ);

    println("Objective functional given control...")
    @btime objectivefunction($cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ);

    println("Optimal policy at a given point...")
    @btime optimalpolicy($t, $Xᵢ, $Vᵢ, $∇Vᵢ);
    @btime optimalpolicygreedy($t, $Xᵢ, $Vᵢ, $∇Vᵢ, $P);
end;

initguess = rand(Float32, 2, 1000)
minimizers = similar(initguess)
for j in axes(initguess, 2)
    minimizers[:, j] .= optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ; c₀ = initguess[:, j])
end

optimalpolicygreedy(t, Xᵢ, Vᵢ, ∇Vᵢ, P)

println("Optimal policy on the full grid...")
policysize = (size(V)..., 2);

policy = Array{Float32, 4}(undef, policysize);
@btime policyovergrid!($policy, $t, $X, $V, $∇V, $P);

# -- Benchmarking G
if false
    ∂ₜV = similar(V);
    w = Array{Float32}(undef, size(V)..., 3);
    ∇V = Array{Float32}(undef, size(V)..., 4);
    @btime G($t, $X, $V, $Ω, $P);
    @btime G!($∂ₜV, $∇V, $w, $policy, $t, $X, $V, $Ω, $P);
end