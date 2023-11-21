using Revise

using Test: @test
using BenchmarkTools
using JLD2
using FiniteDiff

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

vfunc(T, m, y) = -2f0 + (y / log(economy.Y₀))^2 - (T / hogg.Tᵖ)^2;
V = [ vfunc(T, m, y) for T ∈ grid.Ω[1],  m ∈ grid.Ω[2], y ∈ grid.Ω[3] ];
 
χ = similar(V);
ẏ = similar(V);

∂V∂T = similar(V); central∂!(∂V∂T, V, grid, 1);
∂V∂y = similar(V); central∂!(∂V∂y, V, grid, 3); # ∂y
∂²V∂T² = similar(V); ∂²!(∂²V∂T², V, grid, 1);

begin
    i = rand(CartesianIndices(grid))
    Xᵢ = @view grid.X[i, :]
    yᵢ = grid.X[i, 3]
    Vᵢ = @view V[i]
    ∂V∂Tᵢ = @view ∂V∂T[i]
    ∂V∂yᵢ = @view ∂V∂y[i]
    ∂²V∂T²ᵢ = @view ∂²V∂T²[i]
    χᵢ = 0.2f0
    tᵢ = 50f0
end;

# Correctness
@test all(FiniteDiff.finite_difference_gradient(x -> vfunc(x[1], x[2], x[3]), Xᵢ) - [∂V∂T[i], 0, ∂V∂y[i]] .< 1f-4)

@test FiniteDiff.finite_difference_derivative(χ -> Model.f(χ, yᵢ, Vᵢ[1], economy), χᵢ) - Model.Y∂f(χᵢ, yᵢ, Vᵢ[1], economy) < 1f-3
@test FiniteDiff.finite_difference_derivative(
    χ -> hjbterminal(χ, Xᵢ, Vᵢ[1], ∂V∂Tᵢ[1], ∂V∂yᵢ[1], ∂²V∂T²ᵢ[1], instance), χᵢ
) - terminalfoc(χᵢ, Xᵢ, Vᵢ[1], ∂V∂yᵢ[1], economy) < 1f-3

# Performance
terminalfoc(χᵢ, Xᵢ, Vᵢ[1], ∂V∂yᵢ[1], economy)
@code_warntype terminalfoc(χᵢ, Xᵢ, Vᵢ[1], ∂V∂yᵢ[1], economy)
@btime terminalfoc($χᵢ, $Xᵢ, $Vᵢ[1], $∂V∂yᵢ[1], $economy)

optimalterminalpolicy(Xᵢ, Vᵢ[1], ∂V∂yᵢ[1], economy)
@code_warntype optimalterminalpolicy(Xᵢ, Vᵢ[1], ∂V∂yᵢ[1], economy)
@btime optimalterminalpolicy($Xᵢ, $Vᵢ[1], $∂V∂yᵢ[1], $economy)

# Test calls inside terminalG!
Tdir = 1
ydir = 3

@code_warntype central∂!(∂V∂y, V, grid, ydir);
@btime central∂!($∂V∂y, $V, $grid, $ydir);

terminalpolicyovergrid!(χ, V, ∂V∂y, grid, economy)
@code_warntype terminalpolicyovergrid!(χ, V, ∂V∂y, grid, economy);
@btime terminalpolicyovergrid!($χ, $V, $∂V∂y, $grid, $economy);

ȳdrift!(ẏ, χ, grid, instance);
@code_warntype ȳdrift!(ẏ, χ, grid, instance);
@btime ȳdrift!($ẏ, $χ, $grid, $instance);

dir∂!(∂V∂T, V, ẏ, grid, Tdir); dir∂!(∂V∂T, V, ẏ, grid, ydir);
@code_warntype dir∂!(∂V∂T, V, ẏ, grid, Tdir);
@code_warntype dir∂!(∂V∂y, V, ẏ, grid, ydir);
@btime dir∂!($∂V∂y, $V, $ẏ, $grid, $ydir);

∂²!(∂V∂T, V, grid, Tdir);
@code_warntype ∂²!(∂V∂T, V, grid, Tdir);
@btime ∂²!($∂V∂T, $V, $grid, $Tdir);

hjbterminal(χᵢ, Xᵢ, Vᵢ[1], ∂V∂Tᵢ[1], ∂V∂yᵢ[1], ∂²V∂T²ᵢ[1], instance)
@code_warntype hjbterminal(χᵢ, Xᵢ, Vᵢ[1], ∂V∂Tᵢ[1], ∂V∂yᵢ[1], ∂²V∂T²ᵢ[1], instance)
@btime hjbterminal($χᵢ, $Xᵢ, $Vᵢ[1], $∂V∂Tᵢ[1], $∂V∂yᵢ[1], $∂²V∂T²ᵢ[1], $instance);

∂ₜV = similar(V);
terminalG!(∂ₜV, V, ∂V∂y, ∂V∂T, ∂²V∂T², χ, ẏ, grid, instance);
@code_warntype terminalG!(∂ₜV, V, ∂V∂y, ∂V∂T, ∂²V∂T², χ, ẏ, grid, instance);
@btime terminalG!($∂ₜV, $V, $∂V∂y, $∂V∂T, $∂²V∂T², $χ, $ẏ, $grid, $instance);