using Distributed

println("Running with $(nprocs()) processes...")

@everywhere begin # Imports
    using JLD2, Printf
    using Interpolations

    include("../src/vfi.jl")
end


@everywhere begin

    m = MendezFarazmand() # Climate model
    l = LinearQuadratic() # Economic model
    
    n₀ = 40 # size of state space n²
    k₀ = 100 # size of action space

    θ = 0.1
    maxrefinementiters = 100
    maxgridsize = 1500
end

p = 40 # size of parameter space
γspace = range(15, 25; length = p)

@everywhere function adaptivevaluefunction(γ, constrained)
    V, E = adapativevaluefunctioniter(
        m, LinearQuadratic(γ = γ), n₀, k₀;
        maxgridsize, maxrefinementiters, θ,
        constrained = constrained, verbose = false)
    
    return Dict("γ" => γ, "V" => V, "E" => E)
end

println("Unconstrained problem...")
unconstrainedsol = pmap(γ -> adaptivevaluefunction(γ, false), γspace)

println("Constrained problem...")
constrained = pmap(γ -> adaptivevaluefunction(γ, true), γspace)

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