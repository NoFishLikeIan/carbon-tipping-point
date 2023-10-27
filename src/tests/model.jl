using Revise

using Test: @test
using BenchmarkTools
using JLD2
using FiniteDiff # To test gradients

using Model: Economy, Hogg, Albedo
using Model: hjb, objective, gradientobjective!, hessianobjective!, optimalpolicy, policyovergrid!

using Utils: makegrid, fromgridtoarray, central∇, ∂²

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
V = [
    (exp(y) / economy.Ȳ)^2 - (exp(m) / hogg.Mᵖ)^2 *  (T / hogg.Tᵖ)^2 
    for T ∈ Ω[1], m ∈ Ω[2], y ∈ Ω[3]
];
V .= 1f-1 .* (V ./ maximum(abs.(V))); # Renormalise V
∇V = central∇(V, Ω);
∂²V = ∂²(V, Ω);

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
@test all(gradienterror .< 1e-2)

H = zeros(Float32, 2, 2);
@btime hessianobjective!($H, $cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $instance, $calibration);

hessianerror = H .- FiniteDiff.finite_difference_hessian(c -> objective(c, t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration), cᵢ)
@test all(hessianerror .< 1e-3)

println("Optimal policy at a given point...")
@btime optimalpolicy($t, $Xᵢ, $Vᵢ, $∇Vᵢ, $instance, $calibration);

policysize = (size(V)..., 2);
policy = Array{Float32, 4}(undef, policysize);
@btime policyovergrid!($policy, $t, $X, $V, $∇V, $instance, $calibration);

# -- Benchmarking G
if false
    ∂ₜV = similar(V);
    w = Array{Float32}(undef, size(V)..., 3);
    ∇V = Array{Float32}(undef, size(V)..., 4);
    @btime G($t, $X, $V, $Ω, $P);
    @btime G!($∂ₜV, $∇V, $w, $policy, $t, $X, $V, $Ω, $P);
end