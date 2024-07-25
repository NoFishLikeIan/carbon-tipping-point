using Model: TippingModel, LevelDamages, EpsteinZin, GrowthDamages, JumpModel, AbstractModel
using Printf: @sprintf
using UnPack: @unpack

const SIMPATHS = Dict(
    TippingModel{LevelDamages, EpsteinZin}  => "simulation/albedo/level",
    TippingModel{GrowthDamages, EpsteinZin} => "simulation/albedo/growth",
    JumpModel{LevelDamages, EpsteinZin}  => "simulation/jump/level",
    JumpModel{GrowthDamages, EpsteinZin}  => "simulation/jump/growth")

function makefilename(model::TippingModel{LevelDamages, EpsteinZin}, G)
    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ = model.hogg
    @unpack λ₁, λ₂ = model.albedo
    @unpack ξ = model.damages

    N = size(G, 1)
    Δλ = λ₁ - λ₂

    filename = @sprintf("N=%i_Δλ=%.2f_ρ=%.5f_θ=%.2f_ψ=%.2f_σ=%.2f_ω=%.5f_ξ=%.6f", N, Δλ, ρ, θ, ψ, σₜ, ωᵣ, ξ)

    return "$(replace(filename, "." => ",")).jld2"
end

function makefilename(model::TippingModel{GrowthDamages, EpsteinZin}, G)
    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ = model.hogg
    @unpack λ₁, λ₂ = model.albedo
    @unpack ξ, υ = model.damages

    N = size(G, 1)
    Δλ = λ₁ - λ₂

    filename = @sprintf("N=%i_Δλ=%.2f_ρ=%.5f_θ=%.2f_ψ=%.2f_σ=%.2f_ω=%.5f_ξ=%.6f_υ=%.3f", N, Δλ, ρ, θ, ψ, σₜ, ωᵣ, ξ, υ)

    return "$(replace(filename, "." => ",")).jld2"
end

function makefilename(model::JumpModel{GrowthDamages, EpsteinZin}, G)
    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ = model.hogg
    @unpack ξ, υ = model.damages

    N = size(G, 1)

    filename = @sprintf("N=%i_ρ=%.5f_θ=%.2f_ψ=%.2f_σ=%.2f_ω=%.5f_ξ=%.6f_υ=%.3f", N, ρ, θ, ψ, σₜ, ωᵣ, ξ, υ)

    return "$(replace(filename, "." => ",")).jld2"
end

function makefilename(model::JumpModel{LevelDamages, EpsteinZin}, G)
    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ = model.hogg
    @unpack ξ = model.damages

    N = size(G, 1)

    filename = @sprintf("N=%i_ρ=%.5f_θ=%.2f_ψ=%.2f_σ=%.2f_ω=%.5f_ξ=%.6f", N, ρ, θ, ψ, σₜ, ωᵣ, ξ)

    return "$(replace(filename, "." => ",")).jld2"
end

function loadterminal(model::AbstractModel, G; kwargs...)
    dropdims.(loadterminal([model], G; kwargs...); dims = 3)
end

function loadterminal(models::AbstractVector{<:AbstractModel}, G; datapath = "data")
    F̄ = Array{Float64}(undef, N, N, length(models))
    policy = similar(F̄)

    for (k, model) ∈ enumerate(models)
        folder = SIMPATHS[typeof(model)]
        filename = makefilename(model, G)
        savepath = joinpath(datapath, "terminal", folder, filename)
        F̄[:, :, k] .= load(savepath, "F̄")
        policy[:, :, k] .= load(savepath, "policy")
    end

    return F̄, policy
end

# FIXME: Update load total for new value function definition
function loadtotal(model, G; kwargs...)     
    first(loadtotal([model], G; kwargs...))
end
function loadtotal(models::AbstractVector{<:AbstractModel}, G; datapath = "data")
    N = size(G, 1)

    output = Tuple{Vector{Float64}, Array{Float64, 4}, Array{Policy, 4}}[]
    
    for (k, model) ∈ enumerate(models)
        folder = SIMPATHS[typeof(model)]
        filename = makefilename(model, G)
        file = jldopen(joinpath(datapath, folder, "total", filename), "r")

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

function getbool(env, key, fallback)
    key ∈ keys(env) ? env[key] == "true" : fallback
end
function getnumber(env, key, fallback)
    if key ∈ keys(env)
        tol = tryparse(Float64, env[key])

        if !isnothing(tol)
            return tol
        end
    end

    return fallback
end