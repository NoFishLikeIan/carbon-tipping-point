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
    T = @view X[[1], :]
    m = @view X[[2], :]

    eref = Ref(economy)
    href = Ref(hogg)

    [
        μ.(T, m, href, Ref(albedo))
        γᵇ.(t) .- α
        economy.ϱ .+ ϕ.(χ, A(t, economy), eref) .- A(t, economy) .* β.(t, ε(t, exp.(m), α), eref) .- δₖ.(T, eref, href)
    ]
end