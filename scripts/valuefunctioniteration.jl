using Distributed

Γ = [1e-3, 1., 5., 12., 20., 100.]
paramspace = length(Γ)
addprocs(paramspace, exeflags="--project")

println("Running with $paramspace procs...")

using JLD2, Printf

@everywhere begin
    using Interpolations

    include("../src/model/climate.jl")
    include("../src/model/economic.jl")

    include("../src/statecostate/optimalpollution.jl")
    
    function constructinterpolation(X, C, V)
        M = reshape(V, length(X), length(C))
        itp = interpolate((X, C), M, Gridded(Linear()))
    
        return extrapolate(itp, Flat())
    end

    
    function valuefunctioniteration(m::MendezFarazmand, l::LinearQuadratic, n::Int64, k::Int64; h = 1e-2, maxiter = 100_000, tol = 1e-2, verbose = true, cmax = 800., xmax = 350.)
        β = exp(-l.ρ * h)
    
        # State space
        X = range(m.xₚ, xmax, length = n)
        C = range(m.cₚ, cmax, length = n + 1)
        Ω = Base.product(X, C) |> collect |> vec # State space
    
        # Action space
        emax = (l.β₀ - l.τ) / l.β₁
        E = range(-emax, emax; length = k)
        
        L = ((s, e) -> h * (u(e, l) - d(s[1], l))).(Ω, E')
        
        Vᵢ = [H(s[1], s[2], 0, 0, m, l) for s ∈ Ω]
        Eᵢ = zeros(length(Ω))
    
        for i ∈ 1:maxiter
            v = constructinterpolation(X, C, Vᵢ)
    
            v′(s, e) = v(s[1] + h * μ(s[1], s[2], m), s[2] + h * (e - m.δ * s[2]))
            Vₑ = L + β * v′.(Ω, E')
            
            optimalpolicy = argmax(Vₑ, dims = 2)
            
            Vᵢ₊₁ = Vₑ[optimalpolicy]
            Eᵢ₊₁ = [E[index[2]] for index ∈ optimalpolicy] 
    
            ε = maximum(abs.(Vᵢ₊₁ - Vᵢ))
    
            verbose && print("$i / $maxiter: ε = $(round(ε, digits = 4))\r")
    
            if ε < tol
                verbose && println("\nDone at iteration $i with ε = $ε\r")
                e = constructinterpolation(X, C, Eᵢ₊₁)
                return v, e
            end
    
            Eᵢ .= Eᵢ₊₁
            Vᵢ .= Vᵢ₊₁
        end
    
        @warn "Value function iteration did not converge (ε = $ε) in $maxiter iterations."
    
        v = constructinterpolation(X, C, Vᵢ)
        e = constructinterpolation(X, C, Eᵢ)
    
        return v, e
    end

    m = MendezFarazmand()

    n = 30 # size of state space n²
    k = 300 # size of action space
    xmax = 299.5
    cmax = nullcline(xmax, m)


    function pvaluefunctioniteration(γ)
        l = LinearQuadratic(γ = γ)
        
        return valuefunctioniteration(m, l, n, k; cmax = cmax, xmax = xmax, verbose = false)
    end
end

simresults = pmap(pvaluefunctioniteration, Γ)

println("Done parallel running...")

data = Dict()
for (j, γ) in enumerate(Γ)
    v, e = simresults[j]
    data[@sprintf("v_%.0f", γ)] = v
    data[@sprintf("e_%.0f", γ)] = e
end

println("Saving...")

filename = "valuefunction.jld2"
simpath = joinpath("data", "sims", filename)
save(simpath, data)

println("...done!")