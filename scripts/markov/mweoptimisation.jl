using StaticArrays
using Base.Threads
using Optimization, OptimizationOptimJL

mutable struct Vec{T} <: FieldVector{2, T}
    x::T
    y::T
end

StaticArrays.similar_type(::Type{<:Vec}, ::Type{T}, s::Size{(2,)}) where T = Vec{T}
Base.similar(::Type{<:Vec}, ::Type{T}) where T = Vec(zero(T), zero(T))

function rosenbrock(v, p)
    (p[1] - v.x)^2 + p[2] * (v.y - v.x^2)^2
end

function optimovergrid!(maximiser::Matrix{V}, pgrid) where {T, V <: Vec{T}}
    fn = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
    lb = Vec{T}(-1, -1)

    @inbounds @threads for idx in CartesianIndices(pgrid)
        pᵢ = pgrid[idx]
        ub = Vec{T}(Inf, 0.5 + pᵢ[1] * pᵢ[2])
        v₀ = maximiser[idx]
        prob = OptimizationProblem(fn, v₀, pᵢ; lb = lb, ub = ub)
        sol = solve(prob, GradientDescent())

        maximiser[idx] .= sol.u
    end

    return maximiser
end

n = 101
pgrid = [ (p₁, p₂) for p₁ in range(0, 1; length = n), p₂ in range(0, 1; length = n) ]
maximiser = [ similar(Vec{Float64}) for idx in CartesianIndices(pgrid) ]

optimovergrid!(maximiser, pgrid)
@benchmark optimovergrid!($maximiser, $pgrid)