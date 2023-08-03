using Lux
using Lux: Chain, Dense, tanh, kaiming_normal
using Optimisers
using Zygote

using Statistics
using Random

rng = Random.MersenneTwister(1234)
Random.seed!(rng, 1234)

# Definte numeric derivative
ϵ = cbrt(eps(Float32))
ϵ⁻¹ = inv(ϵ)
∂(f, x) = (f(x .+ ϵ) .- f(x .- ϵ)) .* (ϵ⁻¹) ./ 2
∂(f, x, v) = (f(x .+ ϵ * v) - f(x .- ϵ * v)) .* (ϵ⁻¹) ./ 2

n = 20 
model = Chain(
    Dense(1, n, tanh),
    Dense(n, n, tanh),
    Dense(n, 2),
)

ps, st = Lux.setup(rng, model)
nn(x, model, ps, st) = model(x, ps, st) |> first

# Test call
x = rand(rng, Float32, 1, 10_000) .* 10f0
model(x, ps, st) |> first
∂(x -> nn(x, model, ps, st), x)

function loss(x, model, ps, st)
    pde = mean(abs2, nn(x, model, ps, st) .- ∂(x -> nn(x, model, ps, st), x))
    bc = mean(abs2, nn([1f0], model, ps, st) .- [Float32(ℯ), Float32(2ℯ)])
    
    return pde + bc, st
end

function makeoptimiser(ps, η)
    opt = Optimisers.ADAM(η)
    return Optimisers.setup(opt, ps)
end

function train(epochs, iterations; n = 1000)
    ps, st = Lux.setup(rng, model)
    iterloss = Float32[]
    optstate = makeoptimiser(ps, 0.01f0)

    for epoch in 1:epochs
        x = rand(rng, Float32, 1, n) .* 2f0
        totalloss = 0f0

        for i in 1:iterations
            (l, st), back = pullback(p -> loss(x, model, p, st), ps)
            gs = back((one(l), nothing, nothing))[1]
            optstate, ps = Optimisers.update(optstate, ps, gs)

            totalloss += l
            push!(iterloss, l)
        end 

        println("Epoch $epoch avg. loss: $(totalloss / iterations)")
    end

    return ps, st, iterloss
end

ps, st, iterloss = train(10, 250, n = 2000)

using Plots
begin # Plot results
    nn(x) = nn(x', model, ps, st)'
    x = range(0f0, 2f0; length = 101)[:, :]
    plot(x, nn(x); xlabel = "\$x\$", ylabel = "\$y\$", label = "\$NN(x)\$")

    scatter!([1f0, 1f0], nn([1f0])')
end