using DotEnv, JLD2

include("climate.jl")
include("economy.jl")
include("functions.jl")

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
const Eᵇpath = calibration[:Eᵇ][3:end]

function Eᵇ(t)
    dt = economy.t₁ / (length(Eᵇpath) - 1)

    div, port = divrem(t, dt)
    idx = clamp(floor(Int, div) + 1, 1, length(Eᵇpath) - 1)

    α = port / dt

    return Eᵇpath[idx] * (1 - α) + Eᵇpath[idx + 1] * α
end
