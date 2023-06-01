    
include("../src/model/climate.jl")
include("../src/model/economic.jl")

include("../src/statecostate/optimalpollution.jl")
include("../src/utils/grid.jl")

function valuefunctioniter(
    m::MendezFarazmand, l::LinearQuadratic, 
    Γ::Grid, Ω, 
    V₀::Vector{<:Real}, E₀::Vector{<:Real}; 
    h₀ = 1 / 20, maxiter = 100_000, 
    vtol = 1e-2, verbose = true)

    h = h₀ # TODO: Make adaptive step size
    β = exp(-l.ρ * h)
    L = ((s, e) -> u(e, l) - d(s[1], l)).(Γ, Ω')
    
    Vᵢ = copy(V₀)
    Eᵢ = copy(E₀)

    for i ∈ 1:maxiter
        v = constructinterpolation(Γ, Vᵢ)

        v′(s, e) = v(s[1] + h * μ(s[1], s[2], m), s[2] + h * (e - m.δ * s[2]))
        Vₑ = h * L + β * v′.(Γ, Ω')
        
        optimalpolicy = argmax(Vₑ, dims = 2)
        
        Vᵢ₊₁ = vec(Vₑ[optimalpolicy])
        Eᵢ₊₁ = vec([Ω[index[2]] for index ∈ optimalpolicy] )
        η = abs.(Vᵢ₊₁ - Vᵢ)

        ε = maximum(η)

        verbose && print("$i / $maxiter: ε = $(round(ε, digits = 4))\r")

        if ε < vtol
            verbose && print("\nDone at iteration $i with ε = $ε\r")
            e = constructinterpolation(Γ, Eᵢ₊₁)
            return Vᵢ₊₁, Eᵢ₊₁, η
        end

        Eᵢ .= Eᵢ₊₁
        Vᵢ .= Vᵢ₊₁
    end

    throw("Value function iteration did not converge in $maxiter iterations.")
end

function adapativevaluefunctioniter(
    m::MendezFarazmand, l::LinearQuadratic, 
    n₀::Int64, k₀::Int64; 
    constrained = false, θ = 0.1, 
    verbose = true, maxgridsize = 161,
    iterationkwargs...)

    emax = (l.β₀ - l.τ) / l.β₁
    emin = constrained ? 0 : -emax
    
    X₀ = range(m.xₚ, 1.05m.xₚ; length = n₀)
    C₀ = range(m.cₚ, nullcline(maximum(X₀), m); length = n₀)

    Γ = Base.product(X₀, C₀) |> collect |> vec # State space
    Ω = range(emin, emax; length = k₀) |> collect # Start with equally space partition

    V = [H(x, c, 0, 0, m, l) for (x, c) ∈ Γ]
    E = zeros(length(Γ))

    while length(Γ) < maxgridsize
        println("\nGrid refinement with $(length(Γ)) nodes...")

        # Run value function iteration
        V′, E′, η = valuefunctioniter(
            m, l, # Model  
            Γ, Ω, # Grid
            V, E; # Initial value function
            verbose = verbose, iterationkwargs...)

        # Interpolate value and policy on old grid
        v = constructinterpolation(Γ, V′)
        e = constructinterpolation(Γ, E′)

        # Update grid and make new value function initial value
        Γ = updateΓ(Γ, η, θ)

        V = [v(x, c) for (x, c) ∈ Γ]
        E = [e(x, c) for (x, c) ∈ Γ]
    end    

    return V, E, Γ
end


V, E, Γ = adapativevaluefunctioniter(
    m, l, n₀, k₀; 
    verbose = true
)
