include("climate.jl")
include("economy.jl")

# Construct the interpolation functions in t ∈ [0, 1]. A bit ugly hard coding but it works.
const γparams = (4.02245065f-3, 2.6250145f-4, 1.7798517501546115f-6)
γᵇ(t::Float32) = γ(t, γparams, 15f0)

const Eᵇᵥ = Float32.([7.253094489195755, 8.932568341644291, 10.796121480629076, 12.95251456335507, 15.023632003580104, 16.557901521544558, 16.672744124792224, 16.122961603375526])

const economy = Economy()
const hogg = Hogg()
const albedo = Albedo()

const X̄ = [hogg.T̄ hogg.N̅ hogg.m̄ economy.ȳ economy.t₁]'
const X̲ = [hogg.T̲ hogg.N̲ hogg.m̲ economy.y̲ 0f0]'

fromunit(X::Matrix{Float32}) = X .* (X̄ .- X̲) .+ X̲
tounit(X::Matrix{Float32}) = (X .- X̲) ./ (X̄ .- X̲)

function Eᵇ(t::Float32)
    dt = economy.t₁ / (length(Eᵇᵥ) - 1)

    div, port = divrem(t, dt)
    idx = clamp(floor(Int, div) + 1, 1, length(Eᵇᵥ) - 1)

    α = port / dt

    Eᵇᵥ[idx] * (1 - α) + Eᵇᵥ[idx + 1] * α
end

function εcomp(t, N, m, α::Matrix{Float32})::Matrix{Float32}
    ε(t, N,  exp.(m) ./ Eᵇ.(t), α)
end

function ε(t, N, MoverE, α::Matrix{Float32})::Matrix{Float32}
    1f0 .- MoverE .* (δₘ.(N, Ref(hogg)) .+ γᵇ.(t) .- α)
end

function Fα(X::Matrix{Float32}, α::Matrix{Float32})::Matrix{Float32}

    N = @view X[[2], :]
    m = @view X[[3], :]
    t = @view X[[5], :]

    MoverE = exp.(m) ./ Eᵇ.(t)


    [
        1f0 
        economy.ωᵣ * A.(t, Ref(economy)) .* MoverE .* ε(N, MoverE, t, α)
    ]

end


function Fχ(X::Matrix{Float32}, χ::Matrix{Float32})::Matrix{Float32}

    A.(X[[5], :], Ref(economy)) .* (economy.κ .* A.(X[[5], :], Ref(economy)) .* (1 .- χ) .- 1f0)

end

function w(X::Matrix{Float32}, α::Matrix{Float32}, χ::Matrix{Float32})::Matrix{Float32}
    T = @view X[[1], :]
    N = @view X[[2], :]
    m = @view X[[3], :]
    t = @view X[[5], :]

    eref = Ref(economy)
    href = Ref(hogg)

    [
        μ.(T, m, href, Ref(albedo))
        δₘ.(N, href)
        γᵇ.(t) - α
        economy.ϱ .+ ϕ.(χ, A.(t, eref), eref) .- A.(t, eref) .* β.(t, εcomp(t, N, m, α), eref) .- δₖ.(T, eref, href)
        ones(Float32, 1, size(X, 2))
    ]
end