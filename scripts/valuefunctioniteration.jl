using DrWatson; @quickactivate "scc-tipping-points"

using Interpolations

using Printf, BenchmarkTools
using JLD2

using Plots
default(size = 600 .* (√2, 1), dpi = 180, margin = 5Plots.mm)

include("../src/model/climate.jl")
include("../src/model/economic.jl")

include("../src/statecostate/optimalpollution.jl")

"""
Construct interpolated function v(x, c) assuming V is a vector of length |Ω|.
"""
function constructinterpolation(X, C, V)
    M = reshape(V, length(X), length(C))
    itp = interpolate((X, C), M, Gridded(Linear()))

    return extrapolate(itp, Flat())
end

function valuefunctioniteration(m::MendezFarazmand, l::LinearQuadratic, n::Int64, k::Int64; h = 1e-1, maxiter = 1_000, tol = 1e-3, verbose = true)
    β = exp(-l.ρ * h)

    # State space
    X = range(m.xₚ, 300., length = n)
    C = range(m.cₚ, 500., length = n + 1)
    Ω = Base.product(X, C) |> collect |> vec # State space

    # Action space
    emax = (l.β₀ - l.τ) / 2l.β₁
    E = range(-emax, emax; length = k)
    
    L = ((s, e) -> h * (u(e, l) - d(s[1], l))).(Ω, E')
    
    Vᵢ = [H(s[1], s[2], 0, 0, m, l) for s ∈ Ω]
    Eᵢ = zeros(length(Ω))

    for i ∈ 1:maxiter
        v = constructinterpolation(X, C, Vᵢ)

        v′(s, e) = v(s[1] + h * μ(s[1], s[2], m), s[2] + h * (e - m.δ * s[2]))
        Vₑ = L + β * v′.(Ω, E')
        
        optimalpolicy = argmax(Vₑ, dims = 2)
        
        Vᵢ₊₁ = Vₑ[optimalpolicy]
        Eᵢ₊₁ = [E[index[2]] for index ∈ optimalpolicy] 

        ε = maximum(abs.(Vᵢ₊₁ - Vᵢ))

        verbose && print("$i / $maxiter: ε = $(@sprintf("%.4f", ε))\r")

        if ε < tol
            verbose && println("\nDone at iteration $i with ε = $ε\r")
            e = constructinterpolation(X, C, Eᵢ₊₁)
            return v, e
        end

        Eᵢ .= Eᵢ₊₁
        Vᵢ .= Vᵢ₊₁
    end

    @warn "Value function iteration did not converge (ε = $ε) in $maxiter iterations."

    v = constructinterpolation(X, C, Vᵢ)
    e = constructinterpolation(X, C, Eᵢ)

    return v, e
end

m = MendezFarazmand()
l = LinearQuadratic(γ = 20.)

n = 20 # size of state space n²
k = 100 # size of action space
v, e = valuefunctioniteration(m, l, n, k; maxiter = 100_000, h = 1e-2)

X = range(m.xₚ, 300., length = 101)
C = range(m.cₚ, 500., length = 101)

valuefig = contourf(
    X, C, (x, c) -> v(x, c);
    ylabel = "Temperature \$x\$", 
    xlabel = "Carbon concentration \$c\$",
    label = nothing, title = "Value function \$v(x, c)\$"
)

# Save
simpath = joinpath("data", "sims", "valuefunction.jld2")
save(simpath, Dict("v" => v, "e" => e))