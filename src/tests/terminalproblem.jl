using Revise

using Test: @test
using BenchmarkTools
using JLD2
using ImageFiltering: BorderArray

using Utils
using Model

includet("../evolution.jl")

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

grid = RegularGrid(terminaldomain);

Vinner = [ -2f0 + (exp(y) / economy.Y₀)^2 - (T / hogg.Tᵖ)^3 for T ∈ grid.Ω[1],  m ∈ grid.Ω[2], y ∈ grid.Ω[3] ];
V = BorderArray(Vinner, paddims(Vinner, 2));

χ = similar(V.inner);
ẏ = similar(V.inner);

∂V∂T = central∂(V, grid, 1);
∂V∂y = central∂(V, grid, 3); # ∂y
∂²V∂T² = ∂²(V, grid, 1);

begin
    i = rand(CartesianIndices(grid))
    Xᵢ = @view grid.X[i, :]
    Vᵢ = @view V[i]
    ∂V∂Tᵢ = @view ∂V∂T[i]
    ∂V∂yᵢ = @view ∂V∂y[i]
    ∂²V∂T²ᵢ = @view ∂²V∂T²[i]
    χᵢ = 0.5f0
end;

@code_warntype terminalfoc(χᵢ, Xᵢ, Vᵢ[1], ∂V∂yᵢ[1], economy)
@btime terminalfoc($χᵢ, $Xᵢ, $Vᵢ[1], $∂V∂yᵢ[1], $economy)

@code_warntype optimalterminalpolicy(Xᵢ, Vᵢ[1], ∂V∂yᵢ[1], economy)
@btime optimalterminalpolicy($Xᵢ, $Vᵢ[1], $∂V∂yᵢ[1], $economy)

# Test calls inside terminalG!
Tdir = 1
ydir = 3

@code_warntype central∂!(∂V∂y, V, grid, ydir);
@btime central∂!($∂V∂y, $V, $grid, $ydir);

@code_warntype terminalpolicyovergrid!(χ, V, ∂V∂y, grid, economy);
@btime terminalpolicyovergrid!($χ, $V, $∂V∂y, $grid, $economy);

@code_warntype ȳdrift!(ẏ, χ, grid, instance);
@btime ȳdrift!($ẏ, $χ, $grid, $instance);

@code_warntype dir∂!(∂V∂T, V, ẏ, grid, Tdir);
@code_warntype dir∂!(∂V∂y, V, ẏ, grid, ydir);
@btime dir∂!($∂V∂y, $V, $ẏ, $grid, $ydir);

@code_warntype ∂²!(∂V∂T, V, grid, Tdir);
@btime ∂²!($∂V∂T, $V, $grid, $Tdir);

@code_warntype hjbterminal(χᵢ, Xᵢ, Vᵢ[1], ∂V∂yᵢ[1], ∂V∂Tᵢ[1], ∂²V∂T²ᵢ[1], instance)
@btime hjbterminal($χᵢ, $Xᵢ, $Vᵢ[1], $∂V∂yᵢ[1], $∂V∂Tᵢ[1], $∂²V∂T²ᵢ[1], $instance)


∂ₜV = similar(V.inner);
@code_warntype terminalG!(∂ₜV, V, ∂V∂y, ∂V∂T, ∂²V∂T², χ, ẏ, grid, instance);
@btime terminalG!($∂ₜV, $V, $∂V∂y, $∂V∂T, $∂²V∂T², $χ, $ẏ, $grid, $instance);