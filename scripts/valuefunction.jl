using UnPack

using Lux
using NNlib
using Optimisers, Zygote

using StatsBase

using Random
rng = Random.MersenneTwister(1234)

include("../src/model/climate.jl")
include("../src/model/economy.jl")
include("../src/model/init.jl")

include("../src/utils/derivatives.jl")
include("../src/utils/nn.jl")

X₀ = reshape(
    [hogg.T₀, hogg.N̲, log(hogg.M₀), economy.y₀, 0f0],
    (5, 1)
) |> tounit

cube = [paddedrange(x₀, x₀ + 5.1f0ϵ) for x₀ in X₀]

const X = Matrix{Float32}(undef, 5, length(first(cube))^5)
for (i, x) in enumerate(Iterators.product(cube...))
    X[:, i] .= x
end

const n = size(X, 1)
const m = n # 2^7

NN = Chain(
    Dense(n, m, relu),
    Dense(m, m, relu),
    BranchLayer(
        Dense(m, 1, Lux.σ),
        Dense(m, 1, Lux.σ),
        Chain(
            Dense(m, 1),
            WrappedFunction(x -> -Lux.softplus(x))
        )
    )
)
    
function lossfn(Θ, st, X, σ²; weights = ones(Float32, 3))
    (α, χ, V), st = NN(X, Θ, st)

    ∇V = ∇V′w(V, w(X, α, χ), Fα(X, α), Fχ(X, χ))
    
    Y = exp.(X[[4], :])
    χY = χ .* Y

    return mean(abs2,
        weights[1] * (∇V[[1], :] .+ f.(χY, V, Ref(economy)) .+ (σ² / 2f0) .* ∂²₁(V)) +
        weights[1] * ∇V[[2], :] +
        weights[3] * (∇V[[3], :] .+ Y .* ∂f_∂c.(χY, V, Ref(economy)))
    ), st
end

function trainstep!(optimiser, Θ, st, σ²)
	(l, st), back = pullback(p -> lossfn(p, st, X, σ²), Θ)
	gs = back((1f0, nothing))[1]
	
	optimiser, Θ = Optimisers.update!(optimiser, Θ, gs)

	return optimiser, Θ, st, l
end

function train(rng::AbstractRNG, nn, σ²; iterations = 100, initialisation = Lux.setup(rng, nn), η = 1f-3)

	Θ, st = deepcopy.(initialisation)
	
	optimiser = Optimisers.setup(Optimisers.Adam(η), Θ)

    losspath = Vector{Float32}(undef, iterations)

	for iter in 1:iterations
		optimiser, Θ, st, l = trainstep!(optimiser, Θ, st, σ²)
		losspath[iter] = l
	end

	return Θ, st, losspath
end


# Hot load and check allocations
Θ₀, st₀ = Lux.setup(rng, NN)
@time NN(X, Θ₀, st₀)
@time lossfn(Θ₀, st₀, X, 1f0)

testoptimiser = Optimisers.setup(Optimisers.Adam(1f-3), Θ₀)
@time trainstep!(testoptimiser, Θ₀, st₀, 1f0)

@time train(rng, nn; iterations = 1, initialisation = (Θ₀, st₀), η = 1f-3)