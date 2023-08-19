using Lux
using Lux: Chain, Dense, tanh_fast, σ
using Optimisers
using Zygote

using StatsBase

using LinearAlgebra

using Statistics
using Random

using Plots

include("../../src/utils/derivatives.jl")
include("../../src/utils/layers.jl")

rng = Random.MersenneTwister(1234)
Random.seed!(rng, 1234)

nn(x, model, ps, st) = model(x, ps, st) |> first

A = 0.5f0
α = 0.36f0
ρ = 0.04f0
γ = 2.f0
δ = 0.05f0

F(k) = A * k^α
F′(k) = α * A / k^(1 - α)
u(c) = (c^(1 - γ) - 1) / (1 - γ)
u′(c) = c^(-γ)

function loss(x, model, ps, st; λ = 1f0)
    q = nn(x, model, ps, st)
    ∇v = ∂(x -> nn(x, model, ps, st), x)[[1], :]
    v = @view q[[1], :]
    c = @view q[[2], :]
    
    hjb = @. ρ * v - u(c) - ∇v * (F(x) - δ * x - c)
    foc = @. u′(c) - ∇v

    aggloss = @. abs2(hjb) + λ * abs2(foc)

    return mean(aggloss), aggloss,  st
end



n = 20
model = Chain(
    Dense(1, n, Lux.tanh),
    Dense(n, n, Lux.tanh),
    Dense(n, n, Lux.tanh),
    EpsteinZinPositiveControl(n, 2)
)

function makeoptimiser(ps; η = 1f-3, β = (0.9f0, 0999f0))
    opt = Optimisers.Adam(η, β)
    return Optimisers.setup(opt, ps)
end

function generateadaptivebatch(bounds, points, batches, lossvec)
    a, b = bounds
    dx = (b - a) / (points - 1)

    w = weights(all(lossvec .≈ 0) ? ones(Float32, points) : lossvec)

    return (
        sample(rng, a:dx:b, w, points)[:, :]'
        for _ in 1:batches
    )
end

function train(
    model, epochs;
    batch = (1000, 10), 
    xbounds = (0f0, 1f0), η = 1f-3, β = (0.99f0, 0.999f0))

    points, batches = batch

    ps, st = Lux.setup(rng, model)
    losspath = Vector{Float32}(undef, epochs * batches)
    
    optstate = makeoptimiser(ps, η = η, β = β)

    lossvec = zeros(Float32, points)

    for epoch in 1:epochs
        stime = time()
        epochloss = 0f0

        sampler = generateadaptivebatch(xbounds, points, batches, lossvec)

        for (iter, x) in enumerate(sampler)
            (l, lvec, st), back = pullback(p -> loss(x, model, p, st), ps)
            gs = back((one(l), nothing, nothing, nothing))[1]
            optstate, ps = Optimisers.update(optstate, ps, gs)

            
            epochloss += l
            lossvec += lvec'
            losspath[batches * (epoch - 1) + iter] = epochloss
        end

        endtime = time()

        println("Epoch $epoch in $(round(endtime - stime, digits = 2)) s., loss: $(epochloss)")
    end

    return ps, st, losspath
end

kmin = 0f0
kmax = 10f0

epochs = 100
points = 3000
batches = 100

x = rand(rng, Float32, 1, points) .* (kmax - kmin) .+ kmin
ps, st = Lux.setup(rng, model)
nn(x, model, ps, st) # Test model call
loss(x, model, ps, st) # Test loss call

ps, st, iterloss = train(model, epochs; batch = (points, batches), xbounds = (kmin, kmax), η = 5f-4)


# ------ Plotting
kseq = range(1f0, 9f0; length = 101)[:, :]

begin # Plot loss
    lossiterfig = plot(log.(iterloss); ylabel = "Logloss", xlabel = "Iteration", legend = false)

    l, lvec, st = loss(kseq', model, ps, st)

    lossspacefig = plot(kseq, lvec'; ylabel = "Total loss", xlabel = "k", legend = false, color = :darkred)

    plot(lossiterfig, lossspacefig; size = 300 .* (2√2, 1), margins = 5Plots.mm)
end


begin # Plot nn
    q = nn(kseq', model, ps, st)'

    plot(kseq, q[:, [1]]; ylabel = "\$v(k)\$", xlabel = "k", yguidefontcolor = :darkred, color = :darkred, label = false)

    plot!(twinx(), kseq, q[:, [2]]; ylabel = "\$c(k)\$",yguidefontcolor = :darkblue, color = :darkblue, label = false)
end