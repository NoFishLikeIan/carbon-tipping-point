using Distributed

println("Running with $(nprocs()) processes...")

@everywhere begin # Imports
    using JLD2, Printf
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

γspace = 10:5:80
p = length(γspace) # size of parameter space

@everywhere function adaptivevaluefunction(γ, constrained; outerverbose = true, verbose = false, kwargs...)
    l = LinearQuadratic(γ = γ)

    label = constrained ? "constrained" : "unconstrained"

    outerverbose && println("Computing $label value function for γ = $γ...")

    V, E, Γ, η = adapativevaluefunctioniter(
        m, l, n₀, k₀;
        maxgridsize, maxrefinementiters, θ,
        constrained = constrained, verbose = verbose,
        maxiters = maxiters, kwargs...)

    ε = maximum(η)
    outerverbose && println("...done $label problem with γ = $γ with error ε = $ε.")
    
    return Dict(:γ => γ, :V => V, :E => E, :Γ => Γ, :constrained => constrained)
end

paramspace = Iterators.product(γspace, [false, true]) |> collect |> vec

@everywhere function pmapfn(params)
    γ, constrained = params
    return adaptivevaluefunction(γ, constrained; verbose = verbose, outerverbose = outerverbose)
end

solution = pmap(pmapfn, paramspace)

outerverbose && println("Saving...")

filename = "valuefunction.jld2"
simpath = joinpath("data", "sims", filename)
save(simpath, Dict( "solution" => solution ))

outerverbose && println("...done!")
exit()
