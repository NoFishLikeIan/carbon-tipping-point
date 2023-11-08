using Revise

using Test: @test
using BenchmarkTools
using JLD2
using ImageFiltering: BorderArray

using Model
using Utils

economy = Model.Economy();
hogg = Model.Hogg();
albedo = Model.Albedo();

instance = (economy, hogg, albedo);
@load "data/calibration.jld2" calibration;

terminaldomain = [
    (hogg.T₀, hogg.T̄, 101), 
    (log(economy.Y̲), log(economy.Ȳ), 51)
];

Ω = Utils.makegrid(terminaldomain);
X = Utils.fromgridtoarray(Ω);

Vinner = [ -2f0 + (exp(y) / economy.Y₀)^2 - (T / hogg.Tᵖ)^3 for T ∈ Ω[1], y ∈ Ω[2] ]
V = BorderArray(Vinner, Utils.paddims(Vinner, 2))

∂ₜV = similar(V.inner);
policy = similar(V.inner);
w = similar(V.inner);
∂yV = similar(V.inner);
∂²yV = similar(V.inner);
direction = 2;

@btime Utils.central∂!($∂yV, $V, $Ω, $direction);
@btime Utils.∂²!($∂²yV, $V, $Ω, $direction);

begin
    i = rand(CartesianIndices(V.inner))
    yᵢ = @view X[i, 2]
    Vᵢ = @view V[i]
    ∂yVᵢ = @view ∂yV[i]
    ∂²yVᵢ = @view ∂²yV[i]
    χᵢ = 0.5f0
end;

@btime Model.terminalfoc($χᵢ, $yᵢ, $Vᵢ, $∂yVᵢ, $economy)
@btime Model.optimalterminalpolicy($yᵢ, $Vᵢ, $∂yVᵢ, $economy)
@btime Model.terminalpolicyovergrid!($policy, $X, $V, $∂yV, $economy)

