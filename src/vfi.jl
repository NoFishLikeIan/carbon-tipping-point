using Interpolations
using StatsBase

include("../src/model/climate.jl")
include("../src/model/economic.jl")

include("../src/statecostate/optimalpollution.jl")
include("../src/utils/grid.jl")

function valuefunctioniter(
    m::MendezFarazmand, l::LinearQuadratic, 
    Γ::Grid, Ω, 
    V₀::Matrix{<:Real}, E₀::Matrix{<:Real}; 
    h = 1e-2, maxiters = 100_000, 
    vtol = 1e-2, verbose = false)

    β = exp(-l.ρ * h)
    Γvec = Base.product(Γ...) |> collect |> vec
    
    L = ((s, e) -> u(e, l) - d(s[1], l)).(Γvec, Ω')
    
    ηᵢ = Inf .* V₀
    Vᵢ = copy(V₀)
    Eᵢ = copy(E₀)

    for iter ∈ 1:maxiters
        v = constructinterpolation(Γ, Vᵢ)

        function v′(s, e)
            if m.σ²ₓ > 0
                v′₊(s, e) = v(s[1] + h * μ(s[1], s[2], m) + √h * m.σ²ₓ, s[2] + h * (e - m.δ * s[2]))
    
                v′₋(s, e) = v(s[1] + h * μ(s[1], s[2], m) - √h * m.σ²ₓ, s[2] + h * (e - m.δ * s[2]))
    
                (v′₋(s, e) + v′₊(s, e)) / 2
            else
                v(s[1] + h * μ(s[1], s[2], m), s[2] + h * (e - m.δ * s[2]))
            end
        end

        Vₑ = h * L + β * v′.(Γvec, Ω')
        
        optimalpolicy = argmax(Vₑ, dims = 2)
        
        Vᵢ₊₁ = unvec(Vₑ[optimalpolicy], Γ)
        Eᵢ₊₁ = unvec([Ω[index[2]] for index ∈ optimalpolicy], Γ)
        ηᵢ = abs.(Vᵢ₊₁ - Vᵢ)

        ε = maximum(ηᵢ)

        verbose && print("$iter / $maxiters: ε = $(round(ε, digits = 4))\r")

        if ε < vtol
            verbose && print("\nDone at iteration $iter with ε = $(round(ε, digits = 4)) \r")
            return Vᵢ₊₁, Eᵢ₊₁, ηᵢ
        end

        Eᵢ .= Eᵢ₊₁
        Vᵢ .= Vᵢ₊₁
    end

    verbose && "Value function iteration did not converge in $maxiters iterations with ε = $(round(maximum(ηᵢ), digits = 4))"

    return Vᵢ, Eᵢ, ηᵢ
end

function adapativevaluefunctioniter(
    m::MendezFarazmand, l::LinearQuadratic, 
    n₀::Int64, k₀::Int64; 
    constrained = false, θ = 0.1,
    maxrefinementiters = 100, maxgridsize = 401,
    verbose = false, 
    iterationkwargs...)

    emax = (l.β₀ - l.τ) / l.β₁
    emin = constrained ? 0 : -emax
    
    X₀ = range(m.xₚ, 310; length = n₀) |> collect 
    C₀ = range(m.cₚ, 2000; length = n₀) |> collect

    Γ = (X₀, C₀) # State space
    Ω = range(emin, emax; length = k₀) |> collect # Start with equally space partition
    η = zeros(length.(Γ)...)

    V = ((x, c) -> H(x, c, 0, 0, m, l)).(X₀, C₀') # Initial value function guess
    E = copy(η)

    verbose && println("--- Starting refinement with $(gridsize(Γ)) states and $(length(Ω)) policies...")

    for refj ∈ 1:maxrefinementiters
        iseveniter = (refj % 2 == 0)

        if iseveniter && (gridsize(Γ) > maxgridsize) 
            verbose && println("--- ...done refinement $(gridsize(Γ)) states and $(length(Ω)) states!")
            return V, E, Γ, η
        end

        # Run value function iteration
        V′, E′, η = valuefunctioniter(
            m, l, # Model  
            Γ, Ω, # Grid
            V, E; # Initial value function
            verbose = verbose, iterationkwargs...)

        verbose && print("\n")

        # Refine state space
        if iseveniter
            Ω′ = refineΩ(Ω, E, θ)
            verbose && println("--- ...policy refinement: $(length(Ω)) ->  $(length(Ω′)) states...")
            
            Ω = Ω′
        else
            Γ′ = refineΓ(Γ, η, θ)
            verbose && println("--- ...grid refinement: $(gridsize(Γ)) ->  $(gridsize(Γ′)) states...")
            
            # Interpolate value and policy on old grid
            v = constructinterpolation(Γ, V′)
            e = constructinterpolation(Γ, E′)
            # Update grid and make new value function initial value
            Γ = Γ′
            V = v.(Γ[1], Γ[2]')
            E = e.(Γ[1], Γ[2]')
        end
    end    

    verbose && @warn "Maximum number of grid refinement iterations reached."

    return V, E, Γ, η
end
