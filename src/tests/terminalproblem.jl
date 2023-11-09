using Revise

using Test: @test
using BenchmarkTools
using JLD2
using ImageFiltering: BorderArray

using Model: Economy, Hogg, Albedo
using Utils: makegrid, fromgridtoarray, paddims
using Utils: central∂, ∂²

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
    (log(hogg.M̲), log(hogg.M̄), 51), 
    (log(economy.Y̲), log(economy.Ȳ), 51)
];

Ω = makegrid(terminaldomain);
X = fromgridtoarray(Ω);

Vinner = [ -2f0 + (exp(y) / economy.Y₀)^2 - (T / hogg.Tᵖ)^3 for T ∈ Ω[1],  m ∈ Ω[2], y ∈ Ω[3] ];
V = BorderArray(Vinner, paddims(Vinner, 2));

χ = similar(V.inner);
ẏ = similar(V.inner);

∂V∂T = central∂(V, Ω, 1);
∂V∂y = central∂(V, Ω, 3); # ∂y
∂²V∂T² = ∂²(V, Ω, 1);

begin
    i = rand(CartesianIndices(V.inner))
    Xᵢ = @view X[i, :]
    Vᵢ = @view V[i]
    ∂V∂Tᵢ = @view ∂V∂T[i]
    ∂V∂yᵢ = @view ∂V∂y[i]
    ∂²V∂T²ᵢ = @view ∂²V∂T²[i]
    χᵢ = 0.5f0
end;

@btime terminalfoc($χᵢ, $Xᵢ, $Vᵢ[1], $∂V∂yᵢ[1], $economy)
@btime optimalterminalpolicy($Xᵢ, $Vᵢ[1], $∂V∂yᵢ[1], $economy)
@btime terminalpolicyovergrid!($χ, $X, $V, $∂V∂y, $economy)

@btime hjbterminal($χᵢ, $Xᵢ, $Vᵢ[1], $∂V∂yᵢ[1], $∂V∂Tᵢ[1], $∂²V∂T²ᵢ[1], $instance)
@btime ȳdrift!($ẏ, $X, $χ, $instance)

dir∂!(∂V∂T, V, ẏ, Ω, 1);
dir∂!(∂V∂y, V, ẏ, Ω, 3);

∂ₜV = similar(V.inner);
# @btime terminalG!($∂ₜV, $∂yV, $∂²yV, $ẏ, $χ, $X, $V, $Ω, $instance);
