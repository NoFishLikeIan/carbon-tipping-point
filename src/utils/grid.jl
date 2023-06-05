Cell = Tuple{T, T} where T <: Real
Grid = Vector{T} where T <: Cell

function neighbours(cell::Cell, Γ::Grid)::Tuple{Grid, Grid}
    xₗ, cₗ = cell
    
    X = first.(filter(p -> p[2] ≈ cₗ, Γ))
    C = last.(filter(p -> p[1] ≈ xₗ, Γ))

    N, M = length(X), length(C)

    xidx = findfirst(x -> x > xₗ, X)
    cidx = findfirst(c -> c > cₗ, C)

    xneighbour = Base.product(X[findneighbour(xidx, N)], [cₗ]) |> collect |> vec  
    cneighbour = Base.product([xₗ], C[findneighbour(cidx, M)]) |> collect |> vec 

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

function denserΓ(Γ::Grid, η::Vector{Float64}, θ::Float64)::Grid
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

    return sort(Γ′)
end

function coarserΓ(Γ::Grid, V::Vector{<:Real}, η::Vector{Float64}, θ::Float64, ctol::Float64)::Tuple{Grid, Vector{Int64}}
    coarseridx = Integer[]
    η̃ = maximum(η)

    for (j, cell) ∈ enumerate(Γ)
        if η[j] ≥ θ * η̃
            push!(coarseridx, j)
            continue
        end

        xⱼ, cⱼ = cell
        idxs = vcat(1:(j - 1), (j + 1):length(Γ))

        ṽⱼ = constructinterpolation(Γ[idxs], V[idxs])(xⱼ, cⱼ)
        if abs(ṽⱼ - V[j]) ≥ ctol 
            push!(coarseridx, j)
        end
    end

    return Γ[coarseridx], coarseridx
end
    

function constructinterpolation(Γ::Grid, V::Vector{<:Real})
    points = vcat(first.(Γ)', last.(Γ)')
    itp = ScatteredInterpolation.interpolate(Shepard(), points, V)
    return (x, c) -> first(evaluate(itp, [x; c]))
end

plotupdate(Γ, Γ′) = plotupdate(Γ, Γ′, ones(length(Γ)))
function plotupdate(Γ, Γ′, η)
    colors = [cgrad(:Reds, [0, maximum(η)])[ηᵢ] for ηᵢ ∈ η]

    xl, xu = extrema(first.(Γ))
    cl, cu = extrema(last.(Γ))

    Δx = xu - xl
    Δc = cu - cl
    xmar = Δx * 0.05
    cmar = Δc * 0.05

    updatefig = plot(
        xlabel = "\$x\$", ylabel = "\$c\$",
        xlims = (xl - xmar, xu + xmar),
        ylims = (cl - cmar, cu + cmar),
        aspect_ratio = Δx / Δc
    )

    scatter!(updatefig, first.(Γ′), last.(Γ′), color = :black, label = nothing)
    scatter!(updatefig, first.(Γ), last.(Γ), color = colors, label = nothing)

    return updatefig
end

function updateΩ(E′::Vector{<:Real}, Ω::Vector{<:Real})
    emap = countmap(E′)
    Ω′ = copy(Ω)

    for (emissions, counter) ∈ emap
        prop = counter / length(E′)

        eupper = findfirst(>(emissions), Ω)
        neighbs = findneighbour(eupper, length(Ω))

        Δe = abs(emissions - Ω[neighbs[1]])
        newpoints = floor(Int64, length(E′) * prop * 0.5)
    
        newgridpoints = range(emissions - Δe, emissions + Δe; length = 2 * (1 + newpoints))[2:end-1] |> collect

        push!(Ω′, newgridpoints...)
    end

    return sort(Ω′)
end