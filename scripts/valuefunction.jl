using UnPack
using StatsBase
using NLsolve, Optim

include("../src/model/init.jl")
include("../src/utils/derivatives.jl")

function torestricted(Vunr)
    reshape(-1f0 * Vunr.^2, 1, length(Vunr))
end

function fromrestricted(V)
    vec(sqrt.(-1f0 .* V)) 
end

σ(c) = reshape(inv.(1 .+ exp.(c)), 1, length(c))
σ⁻¹(c) = vec(log.(c ./ (1 .- c)))

"""
Given an hypercube grid X ∈ (d, n) and an initial guess for (V₀, χ₀, α₀) ∈ (1, n); compute V, α, χ
"""
function solveincube!(X::Matrix{Float32}, V₀, χ₀, α₀; focweight = 1f0, verbose = false, maxiters = 1000, solver = LBFGS())

    # Allocate constants
    Y = exp.(X[[4], :])
    
    function G(V, χ, α)
        w = drift(X, α, χ)

        D = dir∇V(V, w)

        f.(χ .* Y, V, Ref(economy)) .+ D[[4], :] + (hogg.σ²ₜ / 2f0) .* ∂²T(V) .+ D[[1], :]
    end

    function F(V, χ, α)        
        D = dir∇V(V, drift(X, α, χ))
        Dm = @view D[[3], :]
        Dy = @view D[[4], :]
    
        mean(abs2, Dm .+ Dy .* Fα(X, α)) + 
        mean(abs2, Y .* ∂f_∂c.(χ .* Y, V, Ref(economy)) + Dy .* Fχ(X, χ) )
    end

    n = size(X, 2)
    s₀ = vcat(V₀, α₀, χ₀)

    function loss(state)
        V = torestricted(state[1:n])
        χ = σ(state[(n + 1):2n])
        α = σ(state[(2n + 1):3n])

        return mean(abs2, G(V, χ, α)) + focweight * F(V, χ, α) 
    end

    sol = optimize(loss, s₀, solver, Optim.Options(show_trace = verbose, iterations = maxiters))

    return sol
end


X₀ = reshape(
    [0f0, hogg.T₀, log(hogg.M₀), economy.y₀],
    (4, 1)
) |> tounit

cube = [range(0, 3ϵ; step = ϵ) |> collect for _ in X₀]

X̃  = Matrix{Float32}(undef, 4, length(first(cube))^4)
for (i, x) in enumerate(Iterators.product(cube...))
    X̃[:, i] .= x
end

Xmin = [0f0, hogg.T₀, log(hogg.M₀), economy.y₀]
Xmax = [1f0, hogg.T₀ + 5f0, log(hogg.M₀ + 20f0), economy.y₀ + 1f0]

X = fromunit(X̃, Xmin, Xmax);

n = size(X, 2)
V₀ = -1f0 .* rand(Float32, 1, n)
α = rand(Float32, 1, n)
χ = rand(Float32, 1, n)

V₀ = randn(Float32, n);
α₀ = randn(Float32, n);
χ₀ = randn(Float32, n);


@time sol = solveincube!(X, V₀, α₀, χ₀; verbose = false, maxiters = 2_000);

state = sol.minimizer

V = torestricted(state[1:n])
χ = σ(state[(n + 1):2n])
α = σ(state[(2n + 1):3n])