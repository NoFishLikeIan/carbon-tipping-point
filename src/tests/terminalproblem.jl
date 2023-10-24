using Test, BenchmarkTools

include("../../src/utils/grids.jl")
include("../../src/utils/derivatives.jl")
include("../../src/model/terminalpde.jl")
include("../../src/model/initialisation.jl")

terminaldomain = [
    (hogg.T₀, hogg.T̄, 101), 
    (log(economy.Y̲), log(economy.Ȳ), 51)
];

Ω = makegrid(terminaldomain);
X = fromgridtoarray(Ω);

V = [ -2f0 + (exp(y) / economy.Y₀)^2 - (T / hogg.Tᵖ)^3 for T ∈ Ω[1], y ∈ Ω[2] ]
∂yV = similar(V);
@btime central∂!(∂yV, V, Ω; direction = 2);

begin
    i = rand(CartesianIndices(V))
    Xᵢ = @view X[i, :]
    Vᵢ = @view V[i]
    ∂yVᵢ = @view ∂yV[i]
    χ₀ = 0.5f0
end;

@btime terfoc($χ₀, $Xᵢ, $Vᵢ, $∂yVᵢ);
@btime optimalterminalpolicy($Xᵢ, $Vᵢ, $∂yVᵢ);

policy = similar(V);
@btime terminalpolicyovergrid!($policy, $X, $V, $∂yV);

∂ₜV = similar(V);
@btime terminalG!($∂ₜV, $∂yV, $w, $policy, $X, $V, $Ω);

