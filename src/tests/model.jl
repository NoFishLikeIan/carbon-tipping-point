using Revise

using Test: @test
using BenchmarkTools
using JLD2
using FiniteDiff # To test gradients

using ImageFiltering: BorderArray

using Model: Economy, Hogg, Albedo
using Model: hjb, objective, gradientobjective!, hessianobjective!, optimalpolicy, policyovergrid!

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

# ---- Benchmarking
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
    idx = rand(CartesianIndices(V))
    Xᵢ = @view X[idx, :]
    Vᵢ = @view V[idx]
    ∇Vᵢ = @view ∇V[idx, :]
    ∂²Vᵢ = @view ∂²V[idx]
    cᵢ = rand(Float32, 2)
end;

println("HJB given control...")
@btime hjb($cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $∂²Vᵢ, $instance, $calibration);

println("Objective functional given control...")
@btime objective($cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $instance, $calibration);

∇ = zeros(Float32, 2);
@btime gradientobjective!($∇, $cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $instance, $calibration);

gradienterror = ∇ .- FiniteDiff.finite_difference_gradient(c -> objective(c, t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration), cᵢ)

H = zeros(Float32, 2, 2);
@btime hessianobjective!($H, $cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $instance, $calibration);

hessianerror = H .- FiniteDiff.finite_difference_hessian(c -> objective(c, t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration), cᵢ)
# @test all(hessianerror .< 1e-3)

println("Optimal policy at a given point...")
@btime optimalpolicy($t, $Xᵢ, $Vᵢ, $∇Vᵢ, $instance, $calibration; c₀ = [0.1f0, 0.8f0]);

policysize = (size(V.inner)..., 2);
inner = ones(Float32, policysize) ./ 2f0;
policy = BorderArray(inner, paddims(inner, 1, (1, 2, 3)));
policyovergrid!(policy, t, X, V, ∇V, instance, calibration);
@btime policyovergrid!($policy, $t, $X, $V, $∇V, $instance, $calibration); # FIXME: Gives error

# -- Benchmarking G
∂ₜV = similar(V.inner);
w = ones(Float32, size(X))

G!(∂ₜV, ∇V, ∂²V, policy, w, t, X, V, Ω, instance, calibration);


if false
    w = Array{Float32}(undef, size(V)..., 3);
    ∇V = Array{Float32}(undef, size(V)..., 4);
    @btime G($t, $X, $V, $Ω, $P);
    @btime G!($∂ₜV, $∇V, $w, $policy, $t, $X, $V, $Ω, $P);
end