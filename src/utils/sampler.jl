using StatsBase: weights, sample
using Random: AbstractRNG

adaptivesampler(rng::AbstractRNG, bound::Bound, lossvec::Vector{Float32}) = adaptivesampler(rng, [bound], lossvec)
function adaptivesampler(rng::AbstractRNG, bounds::Bounds, lossvec::Vector{Float32})
    dim = length(bounds)
    points = length(lossvec)

    scale = last.(bounds) - first.(bounds)
    dx = scale ./ (points - 1)

    w = weights(all(lossvec .≈ 0) ? ones(Float32, points) : lossvec)

    X = Matrix{Float32}(undef, dim, points)
    for (i, (a, b)) in enumerate(bounds)
        X[i, :] .= sample(rng, a:dx[i]:b, w, points) 
    end

    return X
end