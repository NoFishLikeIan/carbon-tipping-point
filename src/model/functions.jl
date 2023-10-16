
function ε(t, M, α)
    1f0 .- M .* (δₘ.(M, Ref(hogg)) .+ γᵇ(t) .- α) ./ (Gtonoverppm * Eᵇ(t))
end

function drift(t, X, α, χ)
    T = @view X[:, :, :, 1]
    m = @view X[:, :, :, 2]

    w = similar(X)

    w[:, :, :, 1] .= μ.(T, m, Ref(hogg), Ref(albedo))

    w[:, :, :, 2] .= γᵇ.(t) .- α
    w[:, :, :, 3] .= economy.ϱ .+ ϕ.(χ, A(t, economy), Ref(economy)) .- A(t, economy) .* β.(t, ε(t, exp.(m), α), Ref(economy)) .- δₖ.(T, Ref(economy), Ref(hogg))

    return w
end

function control(χ, α, t, Xᵢ, Vᵢ, ∇Vᵢ)
    f(χ, Xᵢ[3], Vᵢ[1], economy) + 
        ∇Vᵢ[2] * (γᵇ(t) - α) + 
        ∇Vᵢ[3] * (
            ϕ(χ,  A(t, economy), economy) -  A(t, economy) * β(t, ε(t, exp(Xᵢ[2]), α), economy)
        )
end

"""
Computes the objective functional 
    f(t, x, χ) + ∂₂V (γ - α) + ∂₃V (ϕ(χ) - A * β(α))
over the whole state space X
"""
function objectivefunctional(χ::Float32, α::Float32, t::Float32, X::VectorGrid, V::FieldGrid, ∇V::VectorGrid)
    objectivefunctional!(
        Array{Float32}(undef, size(V)), 
        χ, α, 
        t, X, V, ∇V
    )
end
function objectivefunctional!(
    objective::FieldGrid, 
    χ::Float32, α::Float32, 
    t::Float32, X::VectorGrid, V::FieldGrid, ∇V::VectorGrid)
    
    @inbounds for idx ∈ CartesianIndices(V)
        Xᵢ = @view X[idx, :]
        ∇Vᵢ = @view ∇V[idx, :]
        Vᵢ = @view V[idx]

        objective[idx] = control(χ, α, t, Xᵢ, Vᵢ, ∇Vᵢ)
    end

    return objective
end

function optimalpolicy!(policy::VectorGrid, objective::FieldGrid, 
    P::Array{Float32, 3}, t::Float32, X::VectorGrid, 
    V::FieldGrid, ∇V::VectorGrid)

    gridsize = size(P)[1:2]
    
    # pre-allocations
    fᵢ = similar(objective);
    maxjdx = similar(Array{Bool}, axes(objective));
    
    objective .= -Inf32
    for idx in CartesianIndices(gridsize)
        c = @view P[idx, :]
        objectivefunctional!(fᵢ, c[1], c[2], t, X, V, ∇V);
        maxjdx .= fᵢ .> objective
        policy[maxjdx, 1] .= c[1]
        policy[maxjdx, 2] .= c[2]
        objective[maxjdx] .= fᵢ[maxjdx]
    end
end