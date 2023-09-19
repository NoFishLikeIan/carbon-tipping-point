include("climate.jl")
include("economy.jl")

# Construct the interpolation functions in t ∈ [0, 1]. A bit ugly hard coding but it works.
const γparams = (4.0245065f-3, 2.6250145f-4, -1.7798518f-6)

γᵇ(t) = γ(t, γparams, -15f0)

const Eᵇᵥ = Float32.([7.253094489195755, 8.932568341644291, 10.796121480629076, 12.95251456335507, 15.023632003580104, 16.557901521544558, 16.672744124792224, 16.122961603375526])

const economy = Economy()
const hogg = Hogg()
const albedo = Albedo()

const X̄ = [economy.t₁ hogg.T̄ hogg.m̄ economy.ȳ]'
const X̲ = [0f0 hogg.T̲ hogg.m̲ economy.y̲]'

function fromunit(X, Xmin, Xmax)
    X .* (Xmax .- Xmin) .+ Xmin
end
fromunit(X) = fromunit(X, X̲, X̄)

function tounit(X, Xmin, Xmax)
    (X - Xmin) ./ (Xmax .- Xmin)
end
tounit(X) = tounit(X, X̲, X̄)

function Eᵇ(t)
    dt = economy.t₁ / (length(Eᵇᵥ) - 1)

    div, port = divrem(t, dt)
    idx = clamp(floor(Int, div) + 1, 1, length(Eᵇᵥ) - 1)

    α = port / dt

    Eᵇᵥ[idx] * (1 - α) + Eᵇᵥ[idx + 1] * α
end

function ε(t, M, α)
    1f0 .- (M ./ Eᵇ.(t)) .* (δₘ.(hogg.n₀ .* M, Ref(hogg)) .+ γᵇ.(t) .- α)
end

function Fα(X, α)
    t = @view X[[1], :]
    m = @view X[[3], :]

    M = exp.(m)
    economy.ωᵣ * A.(t, Ref(economy)) .* (M ./ Eᵇ.(t)) .* ε(t, M, α)
end


function Fχ(X, χ)
    t = @view X[[4], :]
    A.(t, Ref(economy)) .* (economy.κ .* A.(t, Ref(economy)) .* (1 .- χ) .- 1f0)

end

function drift(X, α, χ)
    t = @view X[[1], :]
    T = @view X[[2], :]
    m = @view X[[3], :]

    eref = Ref(economy)
    href = Ref(hogg)

    [
        μ.(T, m, href, Ref(albedo))
        γᵇ.(t) - α
        economy.ϱ .+ ϕ.(χ, A.(t, eref), eref) .- A.(t, eref) .* β.(t, ε(t, exp.(m), α), eref) .- δₖ.(T, eref, href)
    ]
end