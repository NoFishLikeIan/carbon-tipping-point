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

include("../timederivative.jl")

terminaldomain = [
    (hogg.T₀, hogg.T̄, 101), 
    (log(economy.Y̲), log(economy.Ȳ), 51)
];

Ω = Utils.makegrid(terminaldomain);
X = Utils.fromgridtoarray(Ω);

V = [ -2f0 + (exp(y) / economy.Y₀)^2 - (T / hogg.Tᵖ)^3 for T ∈ Ω[1], y ∈ Ω[2] ]

∂ₜV = similar(V);
policy = similar(V);
w = similar(V);
∂yV = similar(V);
∂²TV = similar(V);

begin
    i = rand(CartesianIndices(V))
    Xᵢ = @view X[i, :]
    Vᵢ = @view V[i]
    ∂yVᵢ = @view ∂yV[i]
    ∂²TVᵢ = @view ∂²TV[i]
    χᵢ = 0.5f0
end;

@btime Model.hjbterminal($χᵢ, $Xᵢ, $Vᵢ, $∂yVᵢ, ∂²TVᵢ, $instance);
@btime Model.terfoc($χᵢ, $Xᵢ, $Vᵢ, $∂yVᵢ, $instance);
@btime Model.optimalterminalpolicy($Xᵢ, $Vᵢ, $∂yVᵢ, instance);

@btime Utils.central∂!(∂yV, V, Ω; direction = 2);
@btime Utils.dir∂!(∂yV, V, w, Ω; direction = 2);

@btime Model.terminalpolicyovergrid!($policy, $X, $V, $∂yV, $instance);

tmp = similar(V, size(V)..., 4);
terminalG!(∂ₜV, tmp, X, V, Ω, instance);
@btime terminalG!($∂ₜV, $tmp, $X, $V, $Ω, $instance);
