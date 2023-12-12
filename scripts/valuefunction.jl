using Model
using JLD2, DotEnv

const env = DotEnv.config()
const DATAPATH = get(env, "DATAPATH", "data")

function getterminal(N, Δλ)::Tuple{Array{Float64, 3}, Array{Float64, 3}, ModelInstance}
    termpath = joinpath(DATAPATH, "terminal", "N=$(N)_Δλ=$(Δλ).jld2")
    @load termpath V̄ policy model
    return V̄, policy, model
end

function computevalue(N::Int, Δλ = 0.08; kwargs...)
    V̄, terminalpolicy, model = getterminal(N, Δλ)

    policy = [Policy(χ, 1e-5) for χ ∈ terminalpolicy]
    V = copy(V̄)

    Model.backwardsimulation!(V, policy, model; kwargs...)
    
    return V, policy, model
end