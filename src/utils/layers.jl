using Lux, NNlib, Random

Bound = Tuple{Float32, Float32}
Bounds = Vector{Bound}

struct BoundedControl <: Lux.AbstractExplicitLayer
    in_dims::Int 
    out_dims::Int
    init_weight::Function
    bounds::Bounds
end

"""
Layer with controls bounded in (c̲, c̄). Expects `bounds = [(c̲₁, c̄₁), (c̲₂, c̄₂)...]`, hence `out_dims = length(bounds) + 1`. Defaults to `bounds = [(0f0, 1f0)...]`.
"""
function BoundedControl(in_dims::Int, out_dims::Int; init_weight::Function = Lux.glorot_normal, bounds::Bounds = repeat([(0f0, 1f0)], out_dims - 1))
    if length(bounds) + 1 != out_dims
        throw(DimensionMismatch("out_dims must be length(bounds) + 1"))
    end

    BoundedControl(in_dims, out_dims, init_weight, bounds)
end

function Lux.initialparameters(rng::AbstractRNG, l::BoundedControl)
    return (
        weight=l.init_weight(rng, l.out_dims, l.in_dims),
    )
end

Lux.initialstates(::AbstractRNG, ::BoundedControl) = NamedTuple()
Lux.parameterlength(l::BoundedControl) = l.out_dims * l.in_dims
Lux.statelength(::BoundedControl) = 0

function (l::BoundedControl)(x::AbstractMatrix, ps, st::NamedTuple)
    y = ps.weight * x

    scale = last.(l.bounds) - first.(l.bounds)
    shift = first.(l.bounds)

    [ y[[1], :]; Lux.σ.(y[2:l.out_dims, :]) .* scale .+ shift ], st
end

struct EpsteinZinBoundedControl <: Lux.AbstractExplicitLayer
    in_dims::Int 
    out_dims::Int
    init_weight::Function
    bounds::Bounds
end

"""
Same as BoundedControl but forces the value function to be negative.
"""
function EpsteinZinBoundedControl(in_dims::Int, out_dims::Int; init_weight::Function = Lux.glorot_normal, bounds::Bounds = repeat([(0f0, 1f0)], out_dims - 1))
    if length(bounds) + 1 != out_dims
        throw(DimensionMismatch("out_dims must be length(bounds) + 1"))
    end

    EpsteinZinBoundedControl(in_dims, out_dims, init_weight, bounds)
end

function Lux.initialparameters(rng::AbstractRNG, l::EpsteinZinBoundedControl)
    return (
        weight=l.init_weight(rng, l.out_dims, l.in_dims),
    )
end

Lux.initialstates(::AbstractRNG, ::EpsteinZinBoundedControl) = NamedTuple()
Lux.parameterlength(l::EpsteinZinBoundedControl) = l.out_dims * l.in_dims
Lux.statelength(::EpsteinZinBoundedControl) = 0

function (l::EpsteinZinBoundedControl)(x::AbstractMatrix, ps, st::NamedTuple)
    y = ps.weight * x

    scale = last.(l.bounds) - first.(l.bounds)
    shift = first.(l.bounds)

    [ -1f0 .* Lux.softplus.(y[[1], :]); Lux.σ.(y[2:l.out_dims, :]) .* scale .+ shift ], st
end


struct EpsteinZinPositiveControl <: Lux.AbstractExplicitLayer
    in_dims::Int 
    out_dims::Int
    init_weight::Function
end

"""
Same as EpsteinZinBoundedControl but forces the value function to be negative and the control to be positive
"""
EpsteinZinPositiveControl(in_dims::Int, out_dims::Int; init_weight::Function = Lux.glorot_normal) = EpsteinZinPositiveControl(in_dims, out_dims, init_weight)

function Lux.initialparameters(rng::AbstractRNG, l::EpsteinZinPositiveControl)
    return (
        weight=l.init_weight(rng, l.out_dims, l.in_dims),
    )
end

Lux.initialstates(::AbstractRNG, ::EpsteinZinPositiveControl) = NamedTuple()
Lux.parameterlength(l::EpsteinZinPositiveControl) = l.out_dims * l.in_dims
Lux.statelength(::EpsteinZinPositiveControl) = 0

function (l::EpsteinZinPositiveControl)(x::AbstractMatrix, ps, st::NamedTuple)
    y = ps.weight * x

    [ -1f0 .* Lux.softplus.(y[[1], :]); Lux.softplus.(y[2:l.out_dims, :])], st
end