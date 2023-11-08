using Revise

using Test: @test
using BenchmarkTools
using JLD2
using ImageFiltering: BorderArray

using Model: Economy, Hogg, Albedo
using Utils: makegrid, fromgridtoarray, central∂, ∂², paddims, dir∂!

# To test...
using Model: terminalpolicyovergrid!, ȳdrift!, terminalfoc, hjbterminal, optimalterminalpolicy

include("../evolution.jl")

economy = Economy();
hogg = Hogg();
albedo = Albedo();

instance = (economy, hogg, albedo);
@load "data/calibration.jld2" calibration;

terminaldomain = [
    (hogg.T₀, hogg.T̄, 101), 
    (log(economy.Y̲), log(economy.Ȳ), 51)
];

Ω = makegrid(terminaldomain);
X = fromgridtoarray(Ω);

Vinner = [ -2f0 + (exp(y) / economy.Y₀)^2 - (T / hogg.Tᵖ)^3 for T ∈ Ω[1], y ∈ Ω[2] ];
V = BorderArray(Vinner, paddims(Vinner, 2));

∂ₜV = similar(V.inner);
χ = similar(V.inner);
ẏ = similar(V.inner);

direction = 2;
∂yV = central∂(V, Ω, direction);
∂²yV = ∂²(V, Ω, direction);

begin
    i = rand(CartesianIndices(V.inner))
    Xᵢ = @view X[i, :]
    yᵢ = @view Xᵢ[2]
    Vᵢ = @view V[i]
    ∂yVᵢ = @view ∂yV[i]
    ∂²yVᵢ = @view ∂²yV[i]
    χᵢ = 0.5f0
end;

@btime terminalfoc($χᵢ, $yᵢ, $Vᵢ, $∂yVᵢ, $economy)
@btime optimalterminalpolicy($yᵢ, $Vᵢ, $∂yVᵢ, $economy)
@btime terminalpolicyovergrid!($χ, $X, $V, $∂yV, $economy)

@btime hjbterminal($χᵢ, $Xᵢ, $Vᵢ, $∂yVᵢ, $∂²yVᵢ, $instance)
@btime ȳdrift!($ẏ, $X, $χ, $instance)

dir∂!(∂yV, V, ẏ, Ω, direction);

@btime terminalG!($∂ₜV, $∂yV, $∂²yV, $ẏ, $χ, $X, $V, $Ω, $instance);