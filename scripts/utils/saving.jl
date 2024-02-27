using Printf: @sprintf
using UnPack: @unpack

function makefilename(model::ModelInstance, G::RegularGrid)
    if !(typeof(model.preferences) <: EpsteinZin)
        throw("Not implemented file saving for non Epstein Zin utilities.")
    end

    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ = model.hogg
    @unpack λ₁, λ₂ = model.albedo

    N = size(G, 1)
    Δλ = λ₁ - λ₂

    filename = @sprintf("N=%i_Δλ=%.2f_ρ=%.5f_θ=%.2f_ψ=%.2f_σ=%.2f_ω=%.5f", N, Δλ, ρ, θ, ψ, σₜ, ωᵣ)

    return "$(replace(filename, "." => ",")).jld2"
end

function makefilename(model::ModelBenchmark, G::RegularGrid)
    if !(typeof(model.preferences) <: EpsteinZin)
        throw("Not implemented file saving for non Epstein Zin utilities.")
    end

    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ = model.hogg

    N = size(G, 1)

    filename = @sprintf("N=%i_ρ=%.5f_θ=%.2f_ψ=%.2f_σ=%.2f_ω=%.5f", N, ρ, θ, ψ, σₜ, ωᵣ)

    return "$(replace(filename, "." => ",")).jld2"
end

function loadterminal(model, G::RegularGrid; kwargs...)
    dropdims.(loadterminal([model], G; kwargs...); dims = 4)
end

function loadterminal(models::AbstractVector, G::RegularGrid; datapath = "data")
    path = joinpath(datapath, "terminal")

    V̄ = Array{Float64}(undef, N, N, N, length(models))
    policy = similar(V̄)

    for (k, model) ∈ enumerate(models)
        filename = joinpath(path, makefilename(model, G))
        V̄[:, :, :, k] .= load(filename, "V̄")
        policy[:, :, :, k] .= load(filename, "policy")
    end

    return V̄, policy
end

function loadtotal(model, G::RegularGrid; kwargs...)     
    first(loadtotal([model], G; kwargs...))
end
function loadtotal(models::AbstractVector, G::RegularGrid; datapath = "data")
    simpath = joinpath(datapath, "total")
    N = size(G, 1)

    output = Tuple{Vector{Float64}, Array{Float64, 4}, Array{Policy, 4}}[]
    
    for (k, model) ∈ enumerate(models)
        filename = makefilename(model, G)
        file = jldopen(joinpath(simpath, filename), "r")

        timesteps = sort(parse.(Float64, keys(file)))

        T = length(timesteps)
        V = Array{Float64, 4}(undef, N, N, N, T)
        policy = Array{Policy, 4}(undef, N, N, N, T)

        for (k, tᵢ) ∈ enumerate(timesteps)
            timekey = string(tᵢ)

            V[:, :, :, k] .= file[timekey]["V"]
            policy[:, :, :, k] .= file[timekey]["policy"]
        end

        push!(output, (timesteps, V, policy))

        close(file)
    end

    return output
end