using DotEnv, JLD2

include("climate.jl")
include("economy.jl")

const env = DotEnv.config();
const DATAPATH = get(env, "DATAPATH", "data/");
const calibration = JLD2.load(joinpath(DATAPATH, "calibration.jld2"))["calibration"]

# Initialise parameters
const economy = Economy();
const hogg = Hogg(); 
const albedo = Albedo(); 


# Construct the interpolation functions in t ∈ [0, 1]. A bit ugly hard coding but it works.
const γparams = Float32.(calibration[:γparameters])
γᵇ(t) = γ(t, γparams, economy.t₀)
const Eᵇpath = Float32.(calibration[:Eᵇ][3:end])

function Eᵇ(t)
    dt = economy.t₁ / (length(Eᵇpath) - 1)

    div, port = divrem(t, dt)
    idx = clamp(floor(Int, div) + 1, 1, length(Eᵇpath) - 1)

    α = port / dt

    return Eᵇpath[idx] * (1 - α) + Eᵇpath[idx + 1] * α
end

function ε(t, M::Float32, α)
    1f0 - M * (δₘ(M, hogg) + γᵇ(t) - α) / (Gtonoverppm * Eᵇ(t))
end
function ε′(t, M)
    M / (Gtonoverppm * Eᵇ(t))
end

function ε(t, M::AbstractArray, α)
    1f0 .- M .* (δₘ.(M, Ref(hogg)) .+ γᵇ(t) .- α) ./ (Gtonoverppm * Eᵇ(t))
end

drift(χ, α, t, X) = drift!(similar(X), χ, α, t, X)
function drift!(w, χ, α, t, X)
    T = @view X[:, :, :, 1]
    m = @view X[:, :, :, 2]

    w[:, :, :, 1] .= μ.(T, m, Ref(hogg), Ref(albedo))
    w[:, :, :, 2] .= γᵇ.(t) .- α
    w[:, :, :, 3] .= economy.ϱ .+ ϕ.(χ, A(t, economy), Ref(economy)) .- A(t, economy) .* β.(t, ε(t, exp.(m), α), Ref(economy)) .- δₖ.(T, Ref(economy), Ref(hogg))

    return w
end