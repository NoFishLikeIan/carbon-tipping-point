using ScatteredInterpolation
using StatsBase

using Plots

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
            verbose && print("\nDone at iteration $i with ε = $(round(ε, digits = 4)) \r")
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
    constrained = false, θ = 0.1, ctol = 1e-3,
    maxrefinementiters = 100, maxgridsize = 401,
    verbose = true, 
    iterationkwargs...)

    emax = (l.β₀ - l.τ) / l.β₁
    emin = constrained ? 0 : -emax
    
    X₀ = range(m.xₚ, 310; length = n₀)
    C₀ = range(m.cₚ, 2000; length = n₀)

    Γ = Base.product(X₀, C₀) |> collect |> vec # State space
    Ω = range(emin, emax; length = k₀) |> collect # Start with equally space partition
    η = zeros(length(Γ))

    V = [H(x, c, 0, 0, m, l) for (x, c) ∈ Γ]
    E = zeros(length(Γ))

    for refj ∈ 1:maxrefinementiters
        if length(Γ) > maxgridsize 
            verbose && println("...done refinement $(length(Γ)) states and $(length(Ω)) states!")
            return V, E, Γ, η
        end

        verbose && println("Starting refinement $refj with $(length(Γ)) states and $(length(Ω)) policies...")

        # Run value function iteration
        V′, E′, η = valuefunctioniter(
            m, l, # Model  
            Γ, Ω, # Grid
            V, E; # Initial value function
            verbose = verbose, iterationkwargs...)

        verbose && print("\n")

        # Coarser step
        Γᶜ, coarseridx = coarserΓ(Γ, V′, η, θ, ctol)

        verbose && println("...removing $(length(Γ) - length(Γᶜ)) states...")
        η = η[coarseridx]

        # Denser step
        Γ′ = denserΓ(Γᶜ, η, θ)

        verbose && println("...adding $(length(Γ′) - length(Γᶜ)) states...")

        Ω′ = updateΩ(E′, Ω)
        verbose && println("...adding $(length(Ω′) - length(Ω)) policies...")
        Ω = Ω′
    
        # Interpolate value and policy on old grid
        v = constructinterpolation(Γ, V′)
        e = constructinterpolation(Γ, E′)

        # Update grid and make new value function initial value
        V = [v(x, c) for (x, c) ∈ Γ′]
        E = [e(x, c) for (x, c) ∈ Γ′]
        Γ = Γ′
    end    

    @warn "Maximum number of grid refinement iterations reached."

    return V, E, Γ
end

m = MendezFarazmand() # Climate model
l = LinearQuadratic(γ = 16.) # Economic model

n₀ = 5 # size of state space n²
k₀ = 10 # size of action space

V, E, Γ = adapativevaluefunctioniter(
    m, l, n₀, k₀; 
    verbose = true, maxgridsize = 500,
    ctol = 0.05, θ = 0.2
)


# Plotting


X₀ = range(m.xₚ, 310; length = n₀)
C₀ = range(m.cₚ, 2000; length = n₀)

Γ₀ = Base.product(X₀, C₀) |> collect |> vec # State space

plotupdate(Γ, Γ₀, ones(length(Γ₀)))

v = constructinterpolation(Γ, V)
X = range(m.xₚ, 310; length = 301)
C = range(m.cₚ, 2000; length = 301)

contourf(C, X, (c, x) -> e(x, c), 
    ylabel = "Temperature", xlabel = "Consumption", 
    title = "Value function", 
    colorbar = :none, 
    fillalpha = 0.5, 
    framestyle = :box
)

plotupdate(Γ, [], V)