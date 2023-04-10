using EconPDEs
using Dierckx

using Plots

include("model/optimalpollution.jl")

N = 101

lowregime = range(0., 1.0 - 1e-3; length = N)
highregime = range(1. + 1e-3, 2.0; length = N)

vguess = OrderedDict(:v => -100 * ones(N))

"""
Computes left and right value function for different tax levels, τ.
"""
function solvedeterministic(m::OptimalPollution)

    leftsol, leftresidual = pdesolve(
        deterministic(m), 
        OrderedDict(:x => lowregime), vguess
    )
    
    rightsol, rightresidual = pdesolve(
        deterministic(m), 
        OrderedDict(:x => highregime), vguess
    )

    @assert leftresidual < 1e-3
    @assert rightresidual < 1e-3

    spllow = Spline1D(lowregime, leftsol[:v])
    splhigh = Spline1D(highregime, rightsol[:v])

    function v(x; ν::Int64 = 0)
        if x ≈ m.x₀ return NaN end
        spl = x < m.x₀ ? spllow : splhigh

        return ν > 0 ? derivative(spl, x; nu = ν) : spl(x)
    end

    return v
end

m = OptimalPollution(τ = 0.5)
v₀ = solvedeterministic(m)

lowsolve = pdesolve(firstordercorrection(m, v₀), OrderedDict(:x => lowregime), vguess)