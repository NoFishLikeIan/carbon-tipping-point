using Test: @test
using BenchmarkTools
using LinearAlgebra

include("../utils/grids.jl")
include("../utils/derivatives.jl")

# Tolerance for first and second derivatives
ε¹ = 1f-3
ε² = 2f-2

begin # Initialise three dimensional cube
    domains = [ (0f0, 1f0, 201), (0f0, 1f0, 201), (0f0, 1f0, 201) ];
    grid = makegrid(domains);
    n = size(grid);
end;

# Generating mock data
Xgrid = Iterators.product(grid...) |> collect;

v(tup) = v(tup[1], tup[2], tup[3]);
v(T, m, y) = T^2 * y^2 + log(m + 1)

∇v(tup) = ∇v(tup[1], tup[2], tup[3]);
∇v(T, m, y) = [2T * y^2, 1 / (m + 1), 2y * T^2]

paddedslice(s) = [(1 + s):(n - s) for n ∈ size(V)] # Index without the edge
begin # Exact derivative
    V = v.(Xgrid);
    V′ = Array{Float32}(undef, size(V)..., 3);
    for idx ∈ CartesianIndices(Xgrid)
        V′[idx, :] .= ∇v(Xgrid[idx])
    end;
end

function absnorm(A, B, order)
    maximum(abs.(A - B)[paddedslice(order)..., :])
end

println("Testing and benchmarking:")

# Central difference
println("--- Central Difference Scheme")
Dcentral = Array{Float32}(undef, size(V)..., length(grid));
central∇!(Dcentral, V, grid);
@btime central∇!($Dcentral, $V, $grid);
centralε = absnorm(Dcentral, V′, 1)
@test centralε < ε¹

# Upwind-downind difference
println("--- Upwind scheme")
w = ones(Float32, size(Dcentral));
D = Array{Float32}(undef, size(V)..., length(grid) + 1);
@btime dir∇!($D, $V, $w, $grid);
dir∇!(D, V, w, grid); 

Dfwd = @view D[:, :, :, 1:3];

errors = abs.(Dfwd - V′)[paddedslice(2)..., :];
fwdε = absnorm(Dfwd, V′, 2)
@test fwdε < ε¹

# Second derivative w.r.t. the first argument
∂²Tv(T, m, y) = 2y^2
∂²TV = similar(V);
for idx ∈ CartesianIndices(V)
    ∂²TV[idx] = ∂²Tv(Xgrid[idx]...)
end

println("--- Second derivative")
D² = similar(V);
@btime ∂²!($D², $V, $grid);
∂²!(D², V, grid); 
T²ε = absnorm(D², ∂²TV, 2);
@test all(T²ε .< ε²)

# Time derivative and Runge-Kutta method
if false
    X = permutedims(collect(reinterpret(reshape, Float32, Xgrid)), (2, 3, 4, 1));
    Y₀ = ones(Float32, size(X, 1), size(X, 2), size(X, 3));

    function G(t, X, Yₜ)
        X₁ = @view X[:, :, :, 1]
        X₂ = @view X[:, :, :, 2]

        -(X₁ .+ X₂) .* tan.(X₁ .+ X₂ .* t) .* Yₜ
    end

    begin # test time
        t, h = 0f0, 1f-1;
        println("--- Derivative function")
        @btime G($t, $X, $Y₀);

        println("--- Runge-Kutta step")
        @btime rkstep!($Y₀, $G, $t, $X; h = $h);
    end;

    function y(t, X) 
        Xₚ = prod(X; dims = 4) 
        dropdims(cos.(Xₚ .* t); dims = 4)
    end

    function simulationerror!(Y::Array{Float32, 3}, timegrid)
        timesteps = length(timegrid) - 1
        h = step(timegrid)
        errors = Vector{Float32}(undef, timesteps)

        for i ∈ 1:timesteps
            t = timegrid[i]
            errors[i] = maximum(abs2, Y - y(t, X))
            rkstep!(Y, G, t, X; h = h)
        end

        return errors
    end

    smalltimegrid = range(0f0, 1f0; length = 3); 
    Ysmall = ones(Float32, size(X, 1), size(X, 2), size(X, 3));
    @btime simulationerror!(Ysmall, smalltimegrid);

    timegrid = range(0f0, 1f0; length = 201);
    Y = ones(Float32, size(X, 1), size(X, 2), size(X, 3));;
    @time errors = simulationerror!(Y, timegrid);
    @test maximum(errors) < ε² / step(timegrid)
end