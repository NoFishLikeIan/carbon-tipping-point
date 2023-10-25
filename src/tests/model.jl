using Revise

using Test: @test
using BenchmarkTools
using JLD2

using Model
using Utils

economy = Model.Economy();
hogg = Model.Hogg();
albedo = Model.Albedo();

instance = (economy, hogg, albedo);
@load "data/calibration.jld2" calibration;

# -- Generate state cube
statedomain = [
    (hogg.T₀, hogg.T̄, 101), 
    (log(hogg.M₀), log(hogg.M̄), 51), 
    (log(economy.Y̲), log(economy.Ȳ), 51)
];
Ω = Utils.makegrid(statedomain);
X = Utils.fromgridtoarray(Ω);

# ---- Benchmarking
begin # Test value function and derivatives
    function vguess(Xᵢ)
        ((exp(Xᵢ[3]) / economy.Ȳ)^2 - (exp(Xᵢ[2]) / hogg.Mᵖ)^2 *  (Xᵢ[1] / hogg.Tᵖ)^2)
    end

    V = Array{Float32}(undef, length.(Ω));
    for I ∈ CartesianIndices(V) V[I] = vguess(X[I, :]) end
    V .= 1f-1 .* (V ./ maximum(abs.(V)))

    ∇V = Utils.central∇(V, Ω);
    ∂²V = Utils.∂²(V, Ω);
end;

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
@btime Model.hjb($cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $∂²Vᵢ, $instance, $calibration);

println("Objective functional given control...")
@btime Model.objective($cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $instance, $calibration);

∇ = zeros(Float32, 2);
@btime Model.gradientobjective!($∇, $cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $instance, $calibration);

H = zeros(Float32, 2, 2);
@btime Model.hessianobjective!($H, $cᵢ, $t, $Xᵢ, $Vᵢ, $∇Vᵢ, $instance, $calibration);

println("Optimal policy at a given point...")
@btime Model.optimalpolicy(t, Xᵢ, Vᵢ, ∇Vᵢ, instance, calibration);

policysize = (size(V)..., 2);
policy = Array{Float32, 4}(undef, policysize);
@btime Model.policyovergrid!($policy, $t, $X, $V, $∇V, $instance, $calibration);

# -- Benchmarking G
if false
    ∂ₜV = similar(V);
    w = Array{Float32}(undef, size(V)..., 3);
    ∇V = Array{Float32}(undef, size(V)..., 4);
    @btime G($t, $X, $V, $Ω, $P);
    @btime G!($∂ₜV, $∇V, $w, $policy, $t, $X, $V, $Ω, $P);
end