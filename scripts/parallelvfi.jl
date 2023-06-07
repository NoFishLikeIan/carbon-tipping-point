using Distributed

println("Running with $(nprocs()) processes...")

@everywhere begin # Imports
    using JLD2, Printf
    using Interpolations

    include("../src/vfi.jl")
end


@everywhere begin

    m = MendezFarazmand() # Climate model
    
    n₀ = 100 # size of state space n²
    k₀ = 100 # size of action space

    outerverbose = true
    verbose = false

    θ = 0.1
    maxrefinementiters = 100
    maxiters = 100_000
    maxgridsize = 1000
end

γspace = 10:5:80
p = length(γspace) # size of parameter space

@everywhere function adaptivevaluefunction(γ; constrained = false, outerverbose = true, verbose = false, kwargs...)
    l = LinearQuadratic(γ = γ)

    outerverbose && println("Computing value function for γ = $γ...")

    V, E, Γ, η = adapativevaluefunctioniter(
        m, l, n₀, k₀;
        maxgridsize, maxrefinementiters, θ,
        constrained = constrained, verbose = verbose,
        maxiters = maxiters, kwargs...)

    ε = maximum(η)
    outerverbose && println("...done γ = $γ with error ε = $ε.")
    
    return Dict("γ" => γ, "V" => V, "E" => E, "Γ" => Γ)
end

outerverbose && println("Unconstrained problem...")
unconstrainedsol = pmap(
    γ -> adaptivevaluefunction(γ; 
        constrained = false, verbose = verbose, outerverbose = outerverbose), 
γspace)

outerverbose && println("Constrained problem...")
constrainedsol = pmap(
    γ -> adaptivevaluefunction(γ; 
        constrained = true, verbose = verbose, outerverbose = outerverbose), 
γspace)

outerverbose && println("Saving...")

filename = "valuefunction.jld2"
simpath = joinpath("data", "sims", filename)
save(simpath, Dict(
    "constrained" => constrainedsol, 
    "unconstrained" => unconstrainedsol
))

outerverbose && println("...done!")
exit()
