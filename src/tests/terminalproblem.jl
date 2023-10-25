using Test, BenchmarkTools

include("../utils/grids.jl")
include("../utils/derivatives.jl")
include("../model/terminalpde.jl")
include("../model/initialisation.jl")

terminaldomain = [
    (hogg.T₀, hogg.T̄, 21), 
    (log(economy.Y̲), log(economy.Ȳ), 21)
];

Ω = makegrid(terminaldomain);
X = fromgridtoarray(Ω);

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

@btime hjbterminal($χ₀, $Xᵢ, $Vᵢ, $∂yVᵢ, ∂²TVᵢ)
@btime terfoc($χ₀, $Xᵢ, $Vᵢ, $∂yVᵢ);
@btime optimalterminalpolicy($Xᵢ, $Vᵢ, $∂yVᵢ);

@btime central∂!(∂yV, V, Ω; direction = 2);
@btime dir∂!(∂yV, V, w, Ω; direction = 2);

@btime terminalpolicyovergrid!($policy, $X, $V, $∂yV);

tmp = similar(V, size(V)..., 4)
@btime terminalG!($∂ₜV, $tmp, $X, $V, $Ω);