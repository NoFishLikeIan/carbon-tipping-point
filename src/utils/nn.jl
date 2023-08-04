using Optimisers: Adam, setup
using Lux: setup as Lsetup
using Zygote: pullback

function makeoptimiser(ps; η)
    opt = Adam(η)
    return setup(opt, ps)
end

nn(x, model, ps, st) = model(x, ps, st) |> first


function train(rng::AbstractRNG, model, epochs;
    batch::Tuple{Integer, Integer} = (1000, 10), 
    bounds::Bounds = repeat([(0f0, 1f0)], model[1].in_dims),
    η = 1f-3)

    points, batches = batch

    ps, st = Lsetup(rng, model)
    optstate = makeoptimiser(ps, η = η)

    losspath = Vector{Float32}(undef, epochs * batches)
    lossvec = zeros(Float32, points)

    for epoch in 1:epochs
        stime = time()
        epochloss = 0f0

        x = adaptivesampler(rng, bounds, lossvec)

        for iter in 1:batches
            (l, iterlossvec, st), back = pullback(p -> loss(x, model, p, st), ps)
            gs = back((one(l), nothing, nothing, nothing))[1]
            optstate, ps = Optimisers.update(optstate, ps, gs)

            lossvec .+= iterlossvec
            epochloss += l
            losspath[batches * (epoch - 1) + iter] = l
        end

        endtime = time()

        println("Epoch $epoch in $(round(endtime - stime, digits = 2)) s., total loss: $(epochloss)")
    end

    return ps, st, losspath
end
