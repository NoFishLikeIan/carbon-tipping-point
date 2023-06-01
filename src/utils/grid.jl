Cell = Tuple{T, T} where T <: Real
Grid = Vector{T} where T <: Cell

function neighbours(cell::Cell, Γ::Grid)::Tuple{Grid, Grid}
    xₗ, cₗ = cell
    
    X = first.(filter(p -> p[2] ≈ cₗ, Γ))
    C = last.(filter(p -> p[1] ≈ xₗ, Γ))

    n, m = length(X), length(C)

    xidx = findfirst(x -> x > xₗ, X)
    cidx = findfirst(c -> c > cₗ, C)

    xneighbour = Base.product(X[findneighbour(xidx, length(X))], [cₗ]) |> collect |> vec  
    cneighbour = Base.product([xₗ], C[findneighbour(cidx, length(C))]) |> collect |> vec 

    return xneighbour, cneighbour
end

function findneighbour(upperidx::Union{Int64, Nothing}, n::Int64)::Vector{Int64}
    if isnothing(upperidx)
        [n - 1]
    elseif upperidx == 2
        [upperidx]
    else
        [upperidx - 2, upperidx]
    end
end

function updateΓ(Γ::Grid, η::Vector{Float64}, θ::Float64)::Grid
    errordict = Dict(Γ .=> η)
    ε = maximum(η)
    Γ′ = copy(Γ)

    for cell ∈ Γ[η .≥ θ * ε]
        xₗ, cₗ = cell
        xneighbours, cneighbours = neighbours(cell, Γ)

        ηˣ = maximum(errordict[node] for node in xneighbours)
        ηᶜ = maximum(errordict[node] for node in cneighbours)
        η̃ = max(ηˣ, ηᶜ)

        if ηˣ ≥ θ * η̃
            for node in xneighbours
                xⁿ = (xₗ + node[1]) / 2
                push!(Γ′, (xⁿ, cₗ))
            end         
        end

        if ηᶜ ≥ θ * η̃
            for node in cneighbours
                cⁿ = (cₗ + node[2]) / 2
                push!(Γ′, (xₗ, cⁿ))
            end
        end
    end

    return Γ′
end

function constructinterpolation(Γ::Grid, V::Vector{<:Real})
    points = vcat(first.(Γ)', last.(Γ)')
    itp = ScatteredInterpolation.interpolate(Shepard(), points, V)
    return (x, c) -> first(evaluate(itp, [x; c]))
end