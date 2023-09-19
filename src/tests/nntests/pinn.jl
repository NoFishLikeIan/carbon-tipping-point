using UnPack

using Lux, Optimisers, Zygote
using Optimization, OptimizationOptimJL
using ComponentArrays, LineSearches
using NNlib

using Statistics
using Random

using Plots: plot, plot!

include("../../../src/utils/derivatives.jl")
include("../../../src/utils/plotting.jl")

rng = Random.MersenneTwister(1234)

xₑ = paddedrange(0f0, 1f0)
idxpad = 3:(length(xₑ) - 2)
x = xₑ[:, idxpad]

depad(x::Matrix{Float32}) = @view x[[1], idxpad]

const model = Chain(
	Dense(1, 24, Lux.relu),
	Dense(24, 24, Lux.relu),
	Dense(24, 24, Lux.relu),
	Dense(24, 24, Lux.relu),
	Dense(24, 24, Lux.relu),
	Dense(24, 1),
    BranchLayer(
        WrappedFunction(depad),
        WrappedFunction(∂)
    )
)

const θ₀, st = Lux.setup(rng, model)
@time model(xₑ, θ₀, st);

function loss(Θ; bcw = 1f0)
    y, y′= first(model(xₑ, Θ, st))
	
	diffeq = mean(abs2, @. y^2 + 1f0 - y′)
    bc = abs2(y[1])

	return diffeq + bcw * bc
end

const losspath = Float32[]
function callback(θ, l)
    push!(losspath, l)
    return false
end

# Hot load
@time loss(θ₀)

adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss(x; bcw = 100f0), adtype)

const ps = ComponentArray{Float32}(θ₀)
optprob = Optimization.OptimizationProblem(optf, ps)

const algo = BFGS(; initial_stepnorm=0.01, linesearch=LineSearches.BackTracking())

res = Optimization.solve(optprob, algo; callback, maxiters=250_000)

begin # Error plot
    plot(log.(losspath); xlabel = "Iteration \$k\$", ylabel = "\$\\log \\mathcal{L}(\\theta_k)\$", linewidth = 1, size = 400 .* (√2, 1), dpi = 600, c = :darkgreen)
end

begin
    (ỹ, _), _ = model(xₑ, res.u, st)
    y = tan.(x)

    plot(x', ỹ'; label = "\$NN(x; \\theta)\$", c = :darkgreen, linewidth = 2.5)
    plot!(x', y'; label = "\$\\tan(x)\$", linewidth = 2, linestyle = :dash, c = :darkred)
end