include("climate.jl")
include("economy.jl")

# Initialise parameters
const economy = Economy()
const hogg = Hogg()
const albedo = Albedo()

# Construct the interpolation functions in t ∈ [0, 1]. A bit ugly hard coding but it works.
const γparams = (4.0245065f-3, 2.6250145f-4, -1.7798518f-6)

γᵇ(t) = γ(t, γparams, economy.t₀)

const Eᵇᵥ = Float32.([7.253094489195755, 8.932568341644291, 10.796121480629076, 12.95251456335507, 15.023632003580104, 16.557901521544558, 16.672744124792224, 16.122961603375526])

function Eᵇ(t)
    dt = economy.t₁ / (length(Eᵇᵥ) - 1)

    div, port = divrem(t, dt)
    idx = clamp(floor(Int, div) + 1, 1, length(Eᵇᵥ) - 1)

    α = port / dt

    Eᵇᵥ[idx] * (1 - α) + Eᵇᵥ[idx + 1] * α
end

const X̲ = [ hogg.T̲, log(hogg.M̲), log(economy.Y̲) ]
const X̅ = [ hogg.T̄, log(hogg.M̄), log(economy.Ȳ) ]

fromunit(Z::Matrix{Float32}, X̲, X̅) = Z .* (X̅ .- X̲) .+ X̲
fromunit(Z::Matrix{Float32}) = fromunit(Z, X̲, X̅)

tounit(X::Matrix{Float32}, X̲, X̅) = (X - X̲) ./ (X̅ .- X̲)
tounit(X::Matrix{Float32}) = tounit(X, X̲, X̅)

expand∂(∇U::Matrix{Float32}, X̲, X̅, economy::Economy) = economy.V̲ .* ∇U ./ (X̅ .- X̲)
expand∂(∇U::Matrix{Float32}) = expand∂(∇U, X̲, X̅, economy)

function abatement(t, X, ∇V)
    m = @view X[[2], :]

    ∂mV = @view ∇V[[2], :]
    ∂yV = @view ∇V[[3], :]
    
    clamp.(
        1 .- (exp.(m) ./ Eᵇ(t)) .* (δₘ.(m, Ref(hogg)) .+ γᵇ.(t) - A(t, economy) .* economy.ωᵣ .* ∂mV ./ ∂yV ),
        1f-3, 1f0 - 1f-3
    )
end

function consumption(t, V, ∇V)
    ∂yV = @view ∇V[[3], :]

    clamp.(
        economy.ρ * (1 - economy.θ) .* V ./ (A.(t, Ref(economy)) .* ∂yV),
        1f-3, 1f0 - 1f-3
    )
end

function ε(t, M, α)
    1f0 .- (M ./ Eᵇ(t)) .* (δₘ.(M, Ref(hogg)) .+ γᵇ(t) .- α)
end

function drift(t, X, α, χ)
    T = @view X[:, :, :, 1]
    m = @view X[:, :, :, 2]

    eref = Ref(economy)
    href = Ref(hogg)
    aref = Ref(albedo)

    w = similar(X)

    w[:, :, :, 1] .= μ.(T, m, href, aref)

    w[:, :, :, 2] .= γᵇ.(t) .- α
    w[:, :, :, 3] .= economy.ϱ .+ ϕ.(χ, A(t, economy), eref) .- A(t, economy) .* β.(t, ε(t, exp.(m), α), eref) .- δₖ.(T, eref, href)

    return w
end

"""
Computes the objective functional 
    f(t, x, χ) + ∂₂V (γ - α) + ∂₃V (ϕ(χ) - A * β(α))
over the whole state space X
"""
function objectivefunctional(t::Float32, V::FieldGrid, ∇V::VectorGrid, X::VectorGrid, P::FieldGrid)
    objectivefunctional!(Array{Float32}(undef, size(V)..., 3), t::Float32, V::FieldGrid, ∇V::VectorGrid, X::VectorGrid, P::FieldGrid)
end
function objectivefunctional!(policy::VectorGrid, t::Float32, V::FieldGrid, ∇V::VectorGrid, X::VectorGrid, P::FieldGrid)
    ∂₂∇V = @view ∇V[:, :, :, 2]
    ∂₃∇V = @view ∇V[:, :, :, 3]

    m = @view X[:, :, :, 2]
    y = @view X[:, :, :, 3]    

    Aₜ = A(t, economy)
    policy[:, :, :, 3] .= -Inf; 
    obj = similar(V)

    for i ∈ axes(P, 1), j ∈ axes(P, 2)
        c = @views P[i, j, :]
        obj .= f.(c[2], y, V, Ref(economy)) .+ 
            ∂₂∇V .* (γᵇ(t) - c[1]) .+ 
            ∂₃∇V .* (ϕ.(c[2], Aₜ, Ref(economy)) .- Aₜ .* β.(t, ε(t, exp.(m), c[1]), Ref(economy)))

        optjdx = obj .> policy[:, :, :, 3];
        policy[optjdx, [3]] .= obj[optjdx]
        policy[optjdx, [1]] .= c[1]
        policy[optjdx, [2]] .= c[2]
    end

    return policy
end