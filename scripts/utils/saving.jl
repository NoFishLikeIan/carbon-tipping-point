function makefilename(model::ModelInstance, G::RegularGrid)
    N = size(G, 1)
    Δλ = round(model.albedo.λ₁ - model.albedo.λ₂; digits = 2)

    return makefilename(N, Δλ, model.preferences)
end

function makefilename(N, Δλ, preferences::CRRA)
    filename = "N=$(N)_Δλ=$(Δλ)_ρ=$(preferences.ρ)_θ=$(preferences.θ)"

    return "$(replace(filename, "." => ",")).jld2"
end

function makefilename(N, Δλ, preferences::LogUtility)
    filename = "N=$(N)_Δλ=$(Δλ)_ρ=$(preferences.ρ)"

    return "$(replace(filename, "." => ",")).jld2"
end

function makefilename(N, Δλ, preferences::EpsteinZin)
    filename = "N=$(N)_Δλ=$(Δλ)_ρ=$(preferences.ρ)_θ=$(preferences.θ)_ψ=$(preferences.ψ)"

    return "$(replace(filename, "." => ",")).jld2"
end

loadterminal(N::Int64, Δλ::Real, p::Preferences; kwargs...) = loadterminal(N, [Δλ], p; kwargs...)
function loadterminal(N::Int64, ΔΛ::AbstractVector{<:Real}, p::Preferences; datapath = "data")
    path = joinpath(datapath, "terminal")

    V̄ = Array{Float64}(undef, N, N, N, length(ΔΛ))
    policy = similar(V̄)

    for (k, Δλ) ∈ enumerate(ΔΛ)
        filename = joinpath(path, makefilename(N, Δλ, p))
        V̄[:, :, :, k] .= load(filename, "V̄")
        policy[:, :, :, k] .= load(filename, "policy")
    end

    filename = joinpath(path, makefilename(N, first(ΔΛ), p))
    G = load(filename, "G")
    model = load(filename, "model")

    return V̄, policy, model, G
end

loadtotal(N::Int64, Δλ::Real, p::Preferences; kwargs...) = loadtotal(N, [Δλ], p; kwargs...)
function loadtotal(N::Int64, ΔΛ::AbstractVector{<:Real}, p::Preferences; datapath = "data")
    _, _, model, G = loadterminal(N, ΔΛ, p; datapath)

    simpath = joinpath(datapath, "total")

    timesteps = range(0., model.economy.t₁; step = 0.25)
    V = Array{Float64}(undef, N, N, N, length(ΔΛ), length(timesteps))
    policy = similar(V, Policy)
    
    for (k, Δλ) ∈ enumerate(ΔΛ)
        filename = joinpath(makefilename(N, Δλ, p))
        file = jldopen(joinpath(simpath, filename), "r")

        for (j, tᵢ) ∈ enumerate(timesteps)
            V[:, :, :, k, j] .= file[string(tᵢ)]["V"]
            policy[:, :, :, k, j] .= file[string(tᵢ)]["policy"]
        end

        close(file)
    end

    return timesteps, V, policy, model, G
end