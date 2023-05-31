using Distributed

println("Running with $(nprocs()) processes...")

@everywhere begin
    using JLD2, Printf
    using Interpolations
    
    include("../src/model/climate.jl")
    include("../src/model/economic.jl")

    include("../src/statecostate/optimalpollution.jl")
end

@everywhere function constructinterpolation(X, C, V)
    M = reshape(V, length(X), length(C))
    itp = interpolate((X, C), M, Gridded(Linear()))

    return extrapolate(itp, Flat())
end

@everywhere function valuefunctioniteration(m::MendezFarazmand, l::LinearQuadratic, n::Int64, emissionspace; h = 1e-2, maxiter = 100_000, tol = 1e-2, verbose = true, cmax = 800., xmax = 350.)
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

@everywhere begin
    m = MendezFarazmand() # Climate model
    l = LinearQuadratic() # Economic model

    n = 40 # size of state space n²
    k = 150 # size of action space

    xmax = 300
    cmax = 1200.

    emax = (l.β₀ - l.τ) / l.β₁
    E = range(-emax, emax; length = k)
    E₊ = range(0, emax; length = k)

    X = range(m.xₚ, xmax, length = n)
    C = range(m.cₚ, cmax, length = n + 1)
end

Γ = range(10, 20; length = 41)
p = length(Γ)

println("Computing value function, parameter space $n × $(n+1) × $p = $(p * n * (n + 1))...")

@everywhere function unconstrainedvalue(γ)
    Vᵧ, Eᵧ = valuefunctioniteration(
        m, LinearQuadratic(γ = γ), n, E; 
        cmax = cmax, xmax = xmax, h = 1e-2, verbose = false
    )

    return Dict(
        "γ" => γ, 
        "V" => reshape(Vᵧ, length(X), length(C)),
        "E" => reshape(Eᵧ, length(X), length(C)), 
    )
end

unconstrainedsol = pmap(unconstrainedvalue, Γ)

println("Computing value function, with action constraint, parameter space $n × $(n+1) × $p = $(p * n * (n + 1))...")

@everywhere function constrainedvalue(γ)
    Vᵧ, Eᵧ = valuefunctioniteration(
        m, LinearQuadratic(γ = γ), n, E₊; 
        cmax = cmax, xmax = xmax, h = 1e-2, verbose = false
    )

    return Dict(
        "γ" => γ, 
        "V" => reshape(Vᵧ, length(X), length(C)),
        "E" => reshape(Eᵧ, length(X), length(C)), 
    )
end

constrainedsol = pmap(constrainedvalue, Γ)

println("Saving...")

filename = "valuefunction.jld2"
simpath = joinpath("data", "sims", filename)
save(simpath, Dict(
    "constrained" => constrainedsol, 
    "unconstrained" => unconstrainedsol,
    "X" => X, "C" => C, "Γ" => Γ
))

println("...done!")
exit()