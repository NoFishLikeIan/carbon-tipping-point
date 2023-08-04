using Lux
using Lux: Chain, Dense
using Optimisers
using Zygote

using LinearAlgebra

using Statistics
using Random
using Plots

include("../../src/model/economy.jl")
include("../../src/model/climate.jl")

include("../../src/utils/derivatives.jl")
include("../../src/utils/layers.jl")
include("../../src/utils/sampler.jl")
include("../../src/utils/nn.jl")

economy = Economy()

rng = Random.MersenneTwister(1234)
Random.seed!(rng, 1234)

# Loss function with (v; χ)' = q = nn(x); ℜ²ˣⁿ ↦ ℜ²ˣⁿ
Δϕ(t, χ) = economy.ϱ + ϕ(χ, A(t, economy), economy) - economy.δₖᵖ
Δϕ(x::Matrix{Float32}, q::Matrix{Float32}) = Δϕ.(x[[1], :], q[[2], :])

ϕ′(t, χ) = ϕ′(χ, A(t, economy), economy)
ϕ′(x::Matrix{Float32}, q::Matrix{Float32}) = ϕ′.(x[[1], :], q[[2], :])

f(y, v, χ) = f(χ * exp(y), v, economy)
f(x::Matrix{Float32}, q::Matrix{Float32}) = f.(x[[2], :], q[[1], :], q[[2], :])

function fᵪ(y, v, χ)
    Y = exp(y)
    return ∂f_∂c(χ * Y, v, economy) * Y
end

fᵪ(x::Matrix{Float32}, q::Matrix{Float32}) = fᵪ.(x[[2], :], q[[1], :], q[[2], :])

function loss(x, model, ps, st; w = 1f0)
    q = nn(x, model, ps, st)
    ∇v = ∇(x -> nn(x, model, ps, st)[[1], :], x)

    ∂ₜv = @view ∇v[[1], :]
    ∂ₛv = @view ∇v[[2], :]

    hjb = f(x, q) .+ ∂ₜv .+ ∂ₛv .* Δϕ(x, q)
    foc = fᵪ(x, q) .+ ∂ₛv .* ϕ′(x, q)

    totloss = @. abs2(hjb + w * foc)
     
    return mean(totloss), vec(totloss), st
end

n = 21
model = Chain(
    Dense(2, n, Lux.tanh),
    Dense(n, n, Lux.tanh),
    Dense(n, n, Lux.tanh),
    EpsteinZinBoundedControl(n, 2)
)

bounds = [
    (-1f0, 100f0), # time 
    (0.f0, 3f0) .* Float32(log(economy.Y₀)) # output
]

epochs = 9
points = 1000
batches = 100

# Test computation  
x = rand(rng, Float32, 2, points)
q = nn(x, model, Lux.setup(rng, model)...)
loss(x, model, Lux.setup(rng, model)...)

ps, st, iterloss = train(
    rng, model, epochs; 
    bounds, batch = (points, batches), 
    η = 1f-3
)

begin # Plot loss
    plot(log.(iterloss); xlabel = "Iteration", ylabel = "Logarithm of loss", label = false)
end

begin # Plot nn
    textrema, yextream = bounds

    tseq = range(textrema..., length = 101)
    yseq = range(yextream...; length = 101)
    
    q̃(t, y) = nn(reshape([t; y], 2, 1), model, ps, st)

    valuefig = contourf(
        tseq, yseq, (t, y) -> q̃(t, y)[1], 
        xlabel = "Time", ylabel = "Output", title = "Value",
        c = :Reds, linewidth = 0
    )

    consumptionfig = contourf(
        tseq, yseq, (t, y) -> q̃(t, y)[2], 
        xlabel = "Time", ylabel = "Output", title = "Consumption",
        c = :viridis, linewidth = 0
    )

    plot(valuefig, consumptionfig; size = 300 .* (2√2, 1), margins = 5Plots.mm)
end