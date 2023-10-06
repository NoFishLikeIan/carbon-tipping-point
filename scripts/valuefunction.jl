using UnPack  
using StatsBase
using NLsolve, Optim
using LinearAlgebra

include("../src/utils/grids.jl")
include("../src/utils/derivatives.jl")
include("../src/model/init.jl")

# -- Generate state cube
const statedomain::Vector{Domain} = [
    (hogg.T₀, hogg.T̄, 60), 
    (log(hogg.M₀), log(hogg.M̄), 60), 
    (log(economy.Y̲), log(economy.Ȳ), 25)
];
const Ω = makeregulargrid(statedomain);
const Nₛ = size(Ω);
const X = fromgridtoarray(Ω);

# -- Generate action square
const actiondomain::Vector{Domain} = [
    (1f-3, 1f0 - 1f-3, 11), (1f-3, 1f0 - 1f-3, 11)
]

const Γ = makeregulargrid(actiondomain);
const Nₐ = size(Γ);
const P = fromgridtoarray(Γ);

G(t::Float32, V::FieldGrid, X::VectorGrid) = G!(copy(V), t, V, X);
function G!(∂ₜV::FieldGrid, t::Float32, V::FieldGrid, X::VectorGrid)
    ∇V = central∇(V, Ω);
    policy = objectivefunctional(t, V, ∇V, X, P);

    α = @views policy[:, :, :, 1]
    χ = @views policy[:, :, :, 2]

    w = drift(t, X, α, χ)
    dir∇V = dir∇(V, w, Ω)

    ∂ₜV .= f.(χ, X[:, :, :, 3] , V, Ref(economy)) .+ 
        dir∇V[:, :, :, 4] .+ 
        ∂²(1, V, Ω) * (hogg.σ²ₜ / 2f0)
    
    return ∂ₜV
end

const timegrid = range(0f0, 5f0; length = 101);
function simulate(timegrid, W₀::FieldGrid; verbose = false)
    h = step(timegrid)
    τ = economy.t₁

    Wpath = Array{Float32}(undef, size(W₀)..., length(timegrid) + 1)
    Wpath[:, :, :, 1] .= W₀

    verbose && println("Running backward simulation...")
    
    for (i, t) ∈ enumerate(timegrid)    
        verbose && print("Iteration $i / $(length(timegrid))\r")
        Wpath[:, :, :, i + 1] .= rkstep(Wpath[:, :, :, i], G, τ - t, X; h = h)
    end
    print("\n")

    return Wpath
end

V̄ = @. X[:, :, :, 3]^2 - 1f5;
Wpath = simulate(timegrid, V̄; verbose = true);