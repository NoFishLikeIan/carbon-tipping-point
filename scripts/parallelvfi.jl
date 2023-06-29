using Distributed

println("Running with $(nprocs()) processes...")

@everywhere begin # Imports
    using UnPack
    using JLD2
    using Dates, Printf
    using Interpolations

    include("../src/vfi.jl")
end


@everywhere begin

    m = MendezFarazmand() # Climate model
    
    n₀ = 20 # size of state space n²
    k₀ = 40 # size of action space

    outerverbose = true # Verbosity outside of parallel loop
    verbose = false # Verbosity inside of parallel loop

    θ = 0.1
    maxrefinementiters = 100
    maxiters  = 50_000
    maxgridsize = 1500
end


@everywhere function adaptivevaluefunction(economy::EconomicModel, climate::MendezFarazmand; outerverbose = true, verbose = false, kwargs...)
    outerverbose && println("-- Computing σ²ₓ = $(climate.σ²ₓ)...")

    V, E, Γ, η, refj = adapativevaluefunctioniter(
        climate, economy, n₀, k₀;
        maxgridsize, maxrefinementiters, θ,
        verbose = verbose,
        maxiters = maxiters, kwargs...)
    
    ε = maximum(η)
    outerverbose && println("-- ...done with σ²ₓ = $(climate.σ²ₓ) with error ε = $ε in $refj refinement iterations.")
    
    return Dict(:economy => economy, :climate => climate, :V => V, :E => E, :Γ => Γ)
end

@everywhere begin
    σspace = 0:0.1:1
    p = length(σspace) # size of parameter space
end

outerverbose && println("Running parallel simulation with $p parameter combinations...")

@everywhere function pmapfn(input)
    index, σ²ₓ = input
    economy = Ramsey()
    climate = MendezFarazmand(σ²ₓ = σ²ₓ)

    outerverbose && println("Iteration $index of $p...")

    return adaptivevaluefunction(economy, climate; verbose = verbose, outerverbose = outerverbose)
end

solution = pmap(pmapfn, enumerate(σspace))

outerverbose && println("Saving...")

filename = "valuefunction_$(now().instant.periods.value).jld2"
simpath = joinpath("data", "sims", filename)
save(simpath, Dict( "solution" => solution, "parameters" => σspace))

outerverbose && println("...done!")
exit()