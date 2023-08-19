using UnPack

using Lux, Optimisers, Zygote
using NNlib

using Statistics
using Random

using Plots: plot, plot!

include("../../src/utils/derivatives.jl")
include("../../src/utils/plotting.jl")

rng = Random.MersenneTwister(1234)

const xₑ = paddedrange(0f0, 1f0)
const idxpad = 3:(length(xₑ) - 2)
const x = xₑ[:, idxpad]

depad(x::Matrix{Float32}) = @view x[[1], idxpad]

nn = Chain(
	Dense(1, 24, Lux.relu),
	Dense(24, 24, Lux.relu),
	Dense(24, 24, Lux.relu),
	Dense(24, 1),
    BranchLayer(
        WrappedFunction(depad),
        WrappedFunction(∂)
    )
)

function loss(nn, Θ, st)
    (y, y′), st = nn(xₑ, Θ, st)
	
	diffeq = mean(abs2, @. y^2 + 1f0 - y′)
    bc = abs2(y[1])

	return diffeq + bc, st 
end

function trainstep!(optimiser, Θ, st)
	(l, st), back = pullback(p -> loss(nn, p, st), Θ)
	gs = back((1f0, nothing))[1]
	
	optimiser, Θ = Optimisers.update!(optimiser, Θ, gs)

	return optimiser, Θ, st, l
end

function train(rng::AbstractRNG, nn; iterations = 100, η = 1f-3)
    Θ, st = Lux.setup(rng, nn)
    optimiser = Optimisers.setup(Optimisers.ADAM(η), Θ)
    
    initlosspath = Vector{Float32}(undef, iterations)

    for iter = 1:iterations
        optimiser, Θ, st, l = trainstep!(optimiser, Θ, st)
        initlosspath[iter] = l
    end

    return Θ, st, initlosspath
end

# Hot load
Θ₀, st₀ = Lux.setup(rng, nn)
@time nn(xₑ, Θ₀, st₀)
@time loss(nn, Θ₀, st₀)
@time trainstep!(Optimisers.setup(Optimisers.ADAM(5f-3), Θ₀), Θ₀, st₀)

Θ, st, initlosspath = train(rng, nn; iterations = 50_000)

begin # Error plot
    plot(log.(initlosspath); xlabel = "Iteration \$k\$", ylabel = "\$\\log \\mathcal{L}(\\theta_k)\$", linewidth = 1, size = 400 .* (√2, 1), dpi = 600, c = :darkgreen)
end

begin
    (ỹ, _), _ = nn(xₑ, Θ, st)
    y = tan.(x)

    plot(x', ỹ'; label = "\$NN(x; \\theta)\$", c = :darkgreen, linewidth = 2.5)
    plot!(x', y'; label = "\$\\tan(x)\$", linewidth = 2, linestyle = :dash, c = :darkred)
end