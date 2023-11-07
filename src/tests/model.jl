using Revise

using Test: @test
using BenchmarkTools, Random
rng = MersenneTwister(123)

using FiniteDiff # To test gradients

using JLD2
using ImageFiltering: BorderArray
using Model: Economy, Hogg, Albedo
using Model: hjb, objective!, optimalpolicy, policyovergrid!

using Utils: makegrid, fromgridtoarray, central∇!, ∂²!, paddims

economy = Economy();
hogg = Hogg();
albedo = Albedo();

instance = (economy, hogg, albedo);
@load "data/calibration.jld2" calibration;

# -- Generate state cube
statedomain = [
    (hogg.T₀, hogg.T̄, 101), 
    (log(hogg.M₀), log(hogg.M̄), 51), 
    (log(economy.Y̲), log(economy.Ȳ), 51)
];
Ω = makegrid(statedomain);
X = fromgridtoarray(Ω);

# -- Benchmarking
# ---- Steady state model
T = @view X[:, :, :, 1];

# ---- Full model
Vdata = [
    (exp(y) / economy.Ȳ)^2 - (exp(m) / hogg.Mᵖ)^2 *  (T / hogg.Tᵖ)^2 
    for T ∈ Ω[1], m ∈ Ω[2], y ∈ Ω[3]
];
V = BorderArray(Vdata, paddims(Vdata, 2));
∇V = Array{Float32}(undef, length.(Ω)..., 4);
central∇!(∇V, V, Ω);

∂²V = Array{Float32}(undef, length.(Ω));
∂²!(∂²V, V, Ω);

# Some sample data
begin
    t = 5f0
    idx = rand(rng, CartesianIndices(V.inner))
    Xᵢ = @view X[idx, :]
    Vᵢ = @view V[idx]
    ∇Vᵢ = @view ∇V[idx, :]
    ∂²Vᵢ = @view ∂²V[idx]
    cᵢ = rand(Float32, 2)
end;

println("HJB given control...")
@btime hjb($cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $∂²Vᵢ, $instance, $calibration);

println("Objective functional given control...")
∇ = zeros(Float32, 2);
H = zeros(Float32, 2, 2);
Z = 0f0;
@btime objective!($Z, $∇, $H, $cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $instance, $calibration);

println("Optimal policy at a given point...")
@btime optimalpolicy($t, $Xᵢ, $Vᵢ, $∇Vᵢ, $instance, $calibration; c₀ = $cᵢ);

policysize = (size(V.inner)..., 2);
inner = ones(Float32, policysize) ./ 2f0;
policy = BorderArray(inner, paddims(inner, 1, (1, 2, 3)));
println("Computing policy over grid")
@btime policyovergrid!($policy, $t, $X, $V, $∇V, $instance, $calibration);