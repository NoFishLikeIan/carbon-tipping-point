using Distributed

println("Running with $(nprocs()) processes...")

@everywhere begin # Imports
    using JLD2, Printf
    using ScatteredInterpolation

    include("../src/vfi.jl")
end


@everywhere begin
    m = MendezFarazmand() # Climate model
    l = LinearQuadratic() # Economic model

    n₀ = 12 # size of state space n²
    k₀ = 50 # size of action space
end


Γ = range(10, 20; length = 41)
p = length(Γ)

println("Computing value function, parameter space $n × $(n+1) × $p = $(p * n * (n + 1))...")

@everywhere function unconstrainedvalue(γ)
    Vᵧ, Eᵧ = valuefunctioniter(
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
    Vᵧ, Eᵧ = valuefunctioniter(
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