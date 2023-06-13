using Distributed

println("Running with $(nprocs()) processes...")

@everywhere begin # Imports
    using JLD2, Printf, Dates
    using Interpolations

    include("../src/vfi.jl")
end


@everywhere begin

    m = MendezFarazmand() # Climate model
    
    n₀ = 40 # size of state space n²
    k₀ = 100 # size of action space

    outerverbose = true
    verbose = false

    θ = 0.1
    maxrefinementiters = 100
    maxiters  = 100_000
    maxgridsize = 5000
end


@everywhere function adaptivevaluefunction(l::LinearQuadratic, m::MendezFarazmand, constrained; outerverbose = true, verbose = false, kwargs...)
    label = constrained ? "constrained" : "unconstrained"

    outerverbose && println("Computing $label value function for γ = $(l.γ) and σ²ₓ = $(m.σ²ₓ)...")

    V, E, Γ, η = adapativevaluefunctioniter(
        m, l, n₀, k₀;
        maxgridsize, maxrefinementiters, θ,
        constrained = constrained, verbose = verbose,
        maxiters = maxiters, kwargs...)
    
    ε = maximum(η)
    outerverbose && println("...done $label problem with γ = $γ and σ²ₓ = $(m.σ²ₓ) with error ε = $ε.")
    
    return Dict(:γ => γ, :V => V, :E => E, :Γ => Γ, :constrained => constrained)
end

γspace = 10:5:40
σspace = [0., 3.]

p = length(γspace) * length(σspace) * 2 # size of parameter space

paramspace = Iterators.product(γspace, [false, true], σspace) |> collect |> vec

@everywhere function pmapfn(params)
    γ, σ²ₓ, constrained = params
    l = LinearQuadratic(γ = γ)
    m = MendezFarazmand(σ²ₓ = σ²ₓ)

    return adaptivevaluefunction(l, m, constrained; verbose = verbose, outerverbose = outerverbose)
end

solution = pmap(pmapfn, paramspace)

outerverbose && println("Saving...")

filename = "valuefunction_$(now().instant.periods.value).jld2"
simpath = joinpath("data", "sims", filename)
save(simpath, Dict( "solution" => solution, "parameters" => paramspace))

outerverbose && println("...done!")
exit()