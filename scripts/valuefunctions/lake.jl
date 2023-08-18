using UnPack

using Lux, Optimisers, Zygote
using NNlib

using Statistics, Random, StatsBase

using ChangePrecision

using Plots: plot, plot!, mm, twinx, default

default(label = false, size = 400 .* (√2, 1), linewidth = 2.)

include("../../src/utils/derivatives.jl")
include("../../src/utils/plotting.jl")

const rng = Random.MersenneTwister(1234)

const b = 6f-1
const c = 5f-1
const ρ = 3f-2
const σ = 5f-2
const σ² = σ^2

const x̲ = 1f-2
const x̅ = 4f0

tounit(x̃) = (x̃ - x̲) / (x̅ - x̲)
fromunit(x) = x̲ + x * (x̅ - x̲)

# Initial conditions
const xₑ = paddedrange(3ϵ, 1f0)
const idxpad = 3:(length(xₑ) - 2)
const x = xₑ[:, idxpad]

const x̃ = @. fromunit(x)
const x̃² = x̃ .^2
const a₀ = @. b * x̃ - x̃² / (1 + x̃²)
const v₀ = @. (log(a₀) - c * x̃²) / ρ

# Neural network definition
depad(x::Matrix{Float32}) = @view x[[1], idxpad]
m = 124

begin
	duallayer = Chain(
		BranchLayer(
			WrappedFunction(depad), 
			WrappedFunction(∂), 
			WrappedFunction(∂²)
		),
		disable_optimizations = true
	)

	valuechain = Chain(
		Dense(m, m, Lux.relu),
		Dense(m, m, Lux.relu),
		Dense(m, 1), duallayer
	)

	actionchain = Chain(
		Dense(m, m, Lux.relu),
		Dense(m, m, Lux.relu),
		Dense(m, 1, Lux.softplus),
		WrappedFunction(depad)
	)

	nn = Chain(
		Dense(1, m, Lux.tanh),
		Dense(m, m, Lux.tanh),
		BranchLayer(valuechain, actionchain)
	)
end

# nn: xₑ (1, n + 4) -> [[v(x), ∂v(x), ∂²v(x)], a(x)] (4, n)
function initloss(nn, Θ, st)
	((v, _, _), a), st = nn(xₑ, Θ, st)
	return mean(abs2.(v - v₀) + abs2.(a - a₀)), st
end

function inittrainstep!(optimiser, Θ, st)
	(l, st), back = pullback(p -> initloss(nn, p, st), Θ)
	gs = back((1f0, nothing))[1]
	
	optimiser, Θ = Optimisers.update!(optimiser, Θ, gs)

	return optimiser, Θ, st, l
end

function inittrain(rng::AbstractRNG, nn; iterations = 100, η = 1f-3)

	Θ₀, st₀ = Lux.setup(rng, nn)
	
	optimiser = Optimisers.setup(Optimisers.Adam(η), Θ₀)

    losspath = Vector{Float32}(undef, iterations)

	for iter in 1:iterations
		optimiser, Θ₀, st₀, l = inittrainstep!(optimiser, Θ₀, st₀)
		losspath[iter] = l
	end

	return Θ₀, st₀, losspath
end
# Hot start loss function
testΘ, testst = Lux.setup(rng, nn)

@time ((v1, v2, v3), a1), st = nn(xₑ, testΘ, testst)
@time initloss(nn, testΘ, testst)
@time inittrainstep!(Optimisers.setup(Optimisers.Adam(1f-3), testΘ), testΘ, testst)

Θ₀, st₀, initlosspath = inittrain(rng, nn; iterations = 10_000, η = 1f-3)

begin # Init loss plot
    lossfig = plot(log.(initlosspath); xlabel = "Iteration \$k\$", ylabel = "\$\\log \\mathcal{L}(\\theta_k)\$", linewidth = 1.)
end

begin # Initial plot
	((v, ∂v, ∂²v), a), st = nn(xₑ, Θ₀, st₀)
	vfig = plot(x', a', c = :darkred, ylabel = "\$a(x; \\Theta)\$", yguidefontcolor = :darkred, ylims = (0, 1), opacity = 0.5)
	plot!(vfig, x', a₀', c = :darkred, linestyle = :dash)
	plot!(twinx(vfig), x', v', c = :darkgreen, ylabel = "\$v(x; \\Theta)\$", yguidefontcolor = :darkgreen, opacity = 0.5)
	plot!(vfig, x', v₀', c = :darkgreen, linestyle = :dash)
end

function u(a)
	@. log(a) - c * x̃²
end

function f(a)
	@. a - b * x̃ + x̃² / (1 + x̃²)
end

const x̃²σ²2⁻¹ = @. x̃² * σ² / 2

# Train routine
function loss(nn, Θ, st)
	((v, ∂v, ∂²v), a), st = nn(xₑ, Θ, st)
	
	hjb = mean(abs2, u(a) .+ ∂v .* f(a) .+ ∂²v .* x̃²σ²2⁻¹ .- ρ .* v)
	foc = mean(abs2, @. inv(a) + ∂v)

	return hjb + foc, st
end

function trainstep!(optimiser, Θ, st)
	(l, st), back = pullback(p -> loss(nn, p, st), Θ)
	gs = back((1f0, nothing))[1]
	
	optimiser, Θ = Optimisers.update!(optimiser, Θ, gs)

	return optimiser, Θ, st, l
end

function train(rng::AbstractRNG, nn; iterations = 100, initialisation = Lux.setup(rng, nn), η = 1f-3)

	Θ, st = deepcopy.(initialisation)
	
	optimiser = Optimisers.setup(Optimisers.Adam(η), Θ)

    losspath = Vector{Float32}(undef, iterations)

	for iter in 1:iterations
		optimiser, Θ, st, l = trainstep!(optimiser, Θ, st)
		losspath[iter] = l
	end

	return Θ, st, losspath
end

# Hot load and check allocations
@time loss(nn, Θ₀, st₀)

testoptimiser = Optimisers.setup(Optimisers.Adam(1f-3), Θ₀)
@time trainstep!(testoptimiser, Θ₀, st₀)

@time train(rng, nn; iterations = 1, initialisation = (Θ₀, st₀), η = 1f-3)

# Parameters
iterations = 100_000

Θ, st, iterloss = train(rng, nn; iterations = iterations, initialisation = (Θ₀, st₀), η = 1f-3)

begin # Loss plot
    lossfig = plot(log.(iterloss); xlabel = "Iteration \$k\$", ylabel = "\$\\log \\mathcal{L}(\\theta_k)\$", linewidth = 0.5, size = 400 .* (√2, 1), dpi = 600, label = false)
end

# Comparing solution

begin
	((v, ∂v, ∂²v), a), st = nn(xₑ, Θ, st)
	plot(x', a', c = :darkred, ylabel = "\$a(x; \\Theta)\$", yguidefontcolor = :darkred, label = false, linewidth = 2.2, ylims = (0, 1))
	plot!(twinx(), x', v', c = :darkgreen, ylabel = "\$v(x; \\Theta)\$", yguidefontcolor = :darkgreen, label = false, linewidth = 2.2)
end