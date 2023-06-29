Grid = Tuple{Vector{T}, Vector{T}} where T <: Real
gridsize(Γ::Grid) = prod(length.(Γ))

function subsequentintervals(idxlist)
    isedge = diff(idxlist) .> 1
    leftedge = [1; findall(isedge) .+ 1] 
    rightedge = [findall(isedge); length(idxlist)]

    return zip(idxlist[leftedge], idxlist[rightedge]) |> collect
end

function refinegrid(Y::Vector{<:Real}, ηʸ::Vector{<:Real}, θ::Float64)
    n = length(Y)
    ηᵤ = quantile(ηʸ, 1 - θ)
    ηₗ = quantile(ηʸ, θ)

    Y′ = Y[ηₗ .< ηʸ .< ηᵤ]

    coarseridx = (1:n)[ηʸ .≤ ηₗ]
    for (l, r) ∈ subsequentintervals(coarseridx)
        if r > l        
            Δ = max(((r - l) + 1) ÷ 2, 2)
            newsubinterval = range(Y[l], Y[r]; length = Δ)
            push!(Y′, newsubinterval...)
        end
    end

    denseridxs = (1:n)[ηʸ .≥ ηᵤ] 

    for (l, r) ∈ subsequentintervals(denseridxs)
        if r > l
            Δ = 2((r - l) + 1) 
            newsubinterval = range(Y[l], Y[r]; length = Δ)
            push!(Y′, newsubinterval...)
        else
            for j ∈ neighbours(l, n)
                push!(Y′,mean([Y[l], Y[j]]))
            end
        end
    end
    
    return sort(Y′)
end

function refineΩ(Ω::Vector{<:Real}, E::Matrix{Float64}, θ::Float64)
    counter = countmap(E)
    ηᵉ = [inv(1 + exp(-get(counter, e, 0))) for e ∈ Ω] |> vec
    return refinegrid(Ω, ηᵉ, θ)
end

function refineΓ(Γ::Grid, η::Matrix{Float64}, θ::Float64)::Grid
    X, C = Γ

    ηˣ = mean.(eachrow(η))
    ηᶜ = mean.(eachcol(η))
    
    return (refinegrid(X, ηˣ, θ), refinegrid(C, ηᶜ, θ))
end

function neighbours(i::Int64, n::Int64)
    if i == 1 return [2] end
    if i == n return [n - 1] end

    return [i - 1, i + 1]
end
    

function constructinterpolation(Γ::Grid, V::Matrix{<:Real})
    X, C = Γ
    itp = interpolate((X, C), V, Gridded(Linear()))
    etp = extrapolate(itp, Line())

    return (x, c) -> etp(x, c)
end

function plotupdate(Γ::Grid, Γ′::Grid; plotkwargs...) 
    plotupdate(Γ, Γ′, ones(length.(Γ)...); plotkwargs...)
end

function plotupdate(Γ::Grid, Γ′::Grid, η::Matrix{<:Real}; beforemarkersize = 4, aftermarkersize = 1, plotkwargs...)
    colors = [cgrad(:Reds, [0, maximum(η)])[ηᵢ] for ηᵢ ∈ vec(η)]

    Γp = Base.product(Γ...) |> collect |> vec
    Γ′p = Base.product(Γ′...) |> collect |> vec

    xl, xu = extrema(first.(Γp))
    cl, cu = extrema(last.(Γp))

    Δx = xu - xl
    Δc = cu - cl
    xmar = Δx * 0.05
    cmar = Δc * 0.05

    updatefig = plot(
        xlabel = "\$x,\$ temperature deviations", ylabel = "\$m,\$ carbon concentration",
        xlims = (xl - xmar, xu + xmar),
        ylims = (cl - cmar, cu + cmar),
        aspect_ratio = Δx / Δc,
        plotkwargs...
    )

    scatter!(updatefig, first.(Γ′p), last.(Γ′p), color = :black, label = "New points", markersize = aftermarkersize)

    scatter!(updatefig, first.(Γp), last.(Γp), color = colors, label = nothing, markersize = beforemarkersize)

    return updatefig
end


unvec(M::Matrix{Float64}, Γ::Grid) = reshape(M, length.(Γ)...)