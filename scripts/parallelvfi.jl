using Distributed

println("Running with $(nprocs()) processes...")

@everywhere begin # Imports
    using JLD2, Printf
    using Interpolations

    include("../src/vfi.jl")
end


@everywhere begin

    m = MendezFarazmand() # Climate model
    
    n₀ = 15 # size of state space n²
    k₀ = 100 # size of action space

    verbose = true

    θ = 0.1
    maxrefinementiters = 100
    maxiters = 5000
    maxgridsize = 1000
end

p = 40 # size of parameter space
γspace = [15.]# range(15, 25; length = p)

@everywhere function adaptivevaluefunction(γ; constrained = false, verbose = false, kwargs...)
    l = LinearQuadratic(γ = γ)

    verbose && println("Computing value function for γ = $γ")

    V, E, Γ = adapativevaluefunctioniter(
        m, l, n₀, k₀;
        maxgridsize, maxrefinementiters, θ,
        constrained = constrained, verbose = verbose,
        maxiters = maxiters, kwargs...)
    
    return Dict("γ" => γ, "V" => V, "E" => E, "Γ" => Γ)
end

println("Unconstrained problem...")
unconstrainedsol = pmap(γ -> adaptivevaluefunction(γ; constrained = false, verbose = verbose), γspace)

println("Constrained problem...")
constrainedsol = pmap(γ -> adaptivevaluefunction(γ; constrained = true, verbose = verbose), γspace)

println("Saving...")

filename = "valuefunction.jld2"
simpath = joinpath("data", "sims", filename)
save(simpath, Dict(
    "constrained" => constrainedsol, 
    "unconstrained" => unconstrainedsol
))

println("...done!")
exit()