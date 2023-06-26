using Distributed

println("Running with $(nprocs()) processes...")

@everywhere begin # Imports
    using JLD2
    using Dates, Printf
    using Interpolations

    include("../src/vfi.jl")
end


@everywhere begin

    m = MendezFarazmand() # Climate model
    
    n₀ = 20 # size of state space n²
    k₀ = 100 # size of action space

    outerverbose = true # Verbosity outside of parallel loop
    verbose = false # Verbosity inside of parallel loop

    θ = 0.1
    maxrefinementiters = 1 # 100
    maxiters  = 100_000
    maxgridsize = 2000
end


@everywhere function adaptivevaluefunction(l::LinearQuadratic, m::MendezFarazmand, constrained::Bool; outerverbose = true, verbose = false, kwargs...)
    label = constrained ? "constrained" : "unconstrained"

    outerverbose && println("-- Computing $label value function for γ = $(l.γ) and σ²ₓ = $(m.σ²ₓ)...")

    V, E, Γ, η = adapativevaluefunctioniter(
        m, l, n₀, k₀;
        maxgridsize, maxrefinementiters, θ,
        constrained = constrained, verbose = verbose,
        maxiters = maxiters, kwargs...)
    
    ε = maximum(η)
    outerverbose && println("-- ...done $label problem with γ = $(l.γ) and σ²ₓ = $(m.σ²ₓ) with error ε = $ε.")
    
    return Dict(:γ => l.γ, :V => V, :E => E, :Γ => Γ, :constrained => constrained)
end

@everywhere begin
    γspace = 22:1:30
    σspace = 0:0.1:1
    constraints = [false, true]
    p = length(γspace) * length(σspace) * length(constraints) # size of parameter space
end

paramspace = Iterators.product(γspace, σspace, constraints) |> collect |> vec

@everywhere function pmapfn(input)
    index, params = input
    γ, σ²ₓ, constrained = params
    l = LinearQuadratic(γ = γ)
    m = MendezFarazmand(σ²ₓ = σ²ₓ)

    outerverbose && println("Iteration $index of $p...")

    return adaptivevaluefunction(l, m, constrained; verbose = verbose, outerverbose = outerverbose)
end

solution = pmap(pmapfn, enumerate(paramspace))

outerverbose && println("Saving...")

filename = "valuefunction_$(now().instant.periods.value).jld2"
simpath = joinpath("data", "sims", filename)
save(simpath, Dict( "solution" => solution, "parameters" => paramspace))

outerverbose && println("...done!")
exit()