using JLD2, Printf
using Interpolations

include("../src/model/climate.jl")
include("../src/model/economic.jl")

include("../src/statecostate/optimalpollution.jl")

function constructinterpolation(X, C, V)
    M = reshape(V, length(X), length(C))
    itp = interpolate((X, C), M, Gridded(Linear()))

    return extrapolate(itp, Flat())
end

function valuefunctioniteration(m::MendezFarazmand, l::LinearQuadratic, n::Int64, emissionspace::Vector{Float64}; h = 1e-2, maxiter = 100_000, tol = 1e-2, verbose = true, cmax = 800., xmax = 350.)
    β = exp(-l.ρ * h)

    # State space
    X = range(m.xₚ, xmax, length = n)
    C = range(m.cₚ, cmax, length = n + 1)
    Ω = Base.product(X, C) |> collect |> vec # State space
    
    L = ((s, e) -> h * (u(e, l) - d(s[1], l))).(Ω, emissionspace')
    
    Vᵢ = [H(s[1], s[2], 0, 0, m, l) for s ∈ Ω]
    Eᵢ = zeros(length(Ω))

    for i ∈ 1:maxiter
        v = constructinterpolation(X, C, Vᵢ)

        v′(s, e) = v(s[1] + h * μ(s[1], s[2], m), s[2] + h * (e - m.δ * s[2]))
        Vₑ = L + β * v′.(Ω, emissionspace')
        
        optimalpolicy = argmax(Vₑ, dims = 2)
        
        Vᵢ₊₁ = Vₑ[optimalpolicy]
        Eᵢ₊₁ = [emissionspace[index[2]] for index ∈ optimalpolicy] 

        ε = maximum(abs.(Vᵢ₊₁ - Vᵢ))

        verbose && print("$i / $maxiter: ε = $(round(ε, digits = 4))\r")

        if ε < tol
            verbose && println("\nDone at iteration $i with ε = $ε\r")
            e = constructinterpolation(X, C, Eᵢ₊₁)
            return Vᵢ₊₁, Eᵢ₊₁
        end

        Eᵢ .= Eᵢ₊₁
        Vᵢ .= Vᵢ₊₁
    end

    @warn "Value function iteration did not converge (ε = $ε) in $maxiter iterations."
    return Vᵢ, Eᵢ
end

function damagevaluefunctioniteration(Γ, n, emissionspace, m::MendezFarazmand; xmax = xmax, cmax = cmax, verbose = true, kwargs...)
    p = length(Γ)
    V = Array{Float64, 3}(undef, n, n + 1, p)
    E = Array{Float64, 3}(undef, n, n + 1, p)

    X = range(m.xₚ, xmax, length = n)
    C = range(m.cₚ, cmax, length = n + 1)

    for (i, γ) ∈ enumerate(Γ) 
        verbose && println("Value function $i / $p for γ = $γ...")

        l = LinearQuadratic(γ = γ)
        Vᵧ, Eᵧ = valuefunctioniteration(m, l, n, emissionspace; cmax = cmax, xmax = xmax, h = 1e-2, verbose = verbose, kwargs...)

        V[:, :, i] = reshape(Vᵧ, n, n + 1)
        E[:, :, i] = reshape(Eᵧ, n, n + 1)
    end
    
    vitp = interpolate((X, C, Γ), V, Gridded(Linear()))
    eitp = interpolate((X, C, Γ), E, Gridded(Linear()))

    vitp, eitp
end

m = MendezFarazmand() # Climate model
l = LinearQuadratic() # Economic model

n = 20 # size of state space n²
k = 100 # size of action space
p = 55

xmax = 299.5
cmax = 800.

emax = (l.β₀ - l.τ) / l.β₁
E = range(-emax, emax; length = k)
E₊ = range(0, emax; length = k)

X = range(m.xₚ, xmax, length = n)
C = range(m.cₚ, cmax, length = n + 1)

Γ = range(5, 60; length = p)

println("Computing value function, parameter space $n × $(n+1) × $p = $(p * n * (n + 1))...")
v, e = damagevaluefunctioniteration(Γ, n, k, E; verbose = false, h = 1e-2)

println("Computing value function, with action constraint, parameter space $n × $(n+1) × $p = $(p * n * (n + 1))...")
v₊, e₊ = damagevaluefunctioniteration(Γ, n, k, E₊; verbose = false, h = 1e-2)

println("Saving...")

filename = "valuefunction.jld2"
simpath = joinpath("data", "sims", filename)
save(simpath, Dict(
    "v" => v, "e" => e, 
    "vp" => v₊, "ep" => e₊,
    "X" => X, "C" => C, "Γ" => Γ
))

println("...done!")