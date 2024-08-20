using Model
using Grid: Policy
using Printf: @sprintf
using UnPack: @unpack
using JLD2: jldopen
using FileIO: load


const SIMPATHS = Dict(
    TippingModel{LevelDamages, EpsteinZin}  => "albedo/level",
    TippingModel{GrowthDamages, EpsteinZin} => "albedo/growth",
    JumpModel{LevelDamages, EpsteinZin}  => "jump/level",
    JumpModel{GrowthDamages, EpsteinZin}  => "jump/growth",

    TippingGameModel{LevelDamages, EpsteinZin}  => "albedo/level",
    TippingGameModel{GrowthDamages, EpsteinZin} => "albedo/growth",
    JumpGameModel{LevelDamages, EpsteinZin}  => "jump/level",
    JumpGameModel{GrowthDamages, EpsteinZin}  => "jump/growth"
)

function makefilename(model::TippingModel{LevelDamages, EpsteinZin}, G)
    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ, σₘ = model.hogg
    @unpack Tᶜ = model.albedo
    @unpack ξ = model.damages

    N = size(G, 1)

    filename = @sprintf("N=%i_Tc=%.2f_ρ=%.5f_θ=%.2f_ψ=%.2f_σT=%.4f_σm=%.4f_ωᵣ=%.5f_ξ=%.6f", N, Tᶜ, ρ, θ, ψ, σₜ, σₘ, ωᵣ, ξ)

    return "$(replace(filename, "." => ",")).jld2"
end

function makefilename(model::TippingModel{GrowthDamages, EpsteinZin}, G)
    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ, σₘ = model.hogg
    @unpack Tᶜ = model.albedo
    @unpack ξ, υ = model.damages

    N = size(G, 1)

    filename = @sprintf("N=%i_Tc=%.2f_ρ=%.5f_θ=%.2f_ψ=%.2f_σT=%.4f_σm=%.4f_ωᵣ=%.5f_ξ=%.6f_υ=%.3f", N, Tᶜ, ρ, θ, ψ, σₜ, σₘ, ωᵣ, ξ, υ)

    return "$(replace(filename, "." => ",")).jld2"
end
function makefilename(model::TippingGameModel{GrowthDamages, EpsteinZin}, G)
    @unpack Tᶜ = model.albedo
    ξh, ξl = getproperty.(model.damages, :ξ)
    υh, υl = getproperty.(model.damages, :υ)

    N = size(G, 1)

    filename = @sprintf("N=%i_Tc=%.2f_ξh=%.6f_ξl=%.6f_υh=%.3f_υl=%.3f", N, Tᶜ, ξh, ξl, υh, υl)

    return "$(replace(filename, "." => ",")).jld2"
end

function makefilename(model::JumpModel{GrowthDamages, EpsteinZin}, G)
    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ, σₘ = model.hogg
    @unpack ξ, υ = model.damages

    N = size(G, 1)

    filename = @sprintf("N=%i_ρ=%.5f_θ=%.2f_ψ=%.2f_σT=%.4f_σm=%.4f_ωᵣ=%.5f_ξ=%.6f_υ=%.3f", N, ρ, θ, ψ, σₜ, σₘ, ωᵣ, ξ, υ)

    return "$(replace(filename, "." => ",")).jld2"
end

function makefilename(model::JumpModel{LevelDamages, EpsteinZin}, G)
    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ, σₘ = model.hogg
    @unpack ξ = model.damages

    N = size(G, 1)

    filename = @sprintf("N=%i_ρ=%.5f_θ=%.2f_ψ=%.2f_σT=%.4f_σm=%.4f_ωᵣ=%.5f_ξ=%.6f", N, ρ, θ, ψ, σₜ, σₘ, ωᵣ, ξ)

    return "$(replace(filename, "." => ",")).jld2"
end

Result = Tuple{Vector{Float64}, Array{Float64, 3}, Array{Policy, 3}}

function loadterminal(model::AbstractModel, G; addpath = "", kwargs...)
    dropdims.(loadterminal([model], G; addpath = [addpath], kwargs...); dims = 3)
end

function loadterminal(models::AbstractVector{<:AbstractModel}, G; datapath = "data/simulation", addpath = repeat([""], length(models)))
    F̄ = Array{Float64}(undef, size(G, 1), size(G, 2), length(models))
    policy = similar(F̄)

    for (k, model) ∈ enumerate(models)
        folder = SIMPATHS[typeof(model)]
        filename = makefilename(model, G)
        savepath = joinpath(datapath, folder, "terminal", addpath[k], filename)
        F̄[:, :, k] .= load(savepath, "F̄")
        policy[:, :, k] .= load(savepath, "policy")
    end

    return F̄, policy
end

function loadtotal(model::AbstractModel, G; kwargs...)     
    first(loadtotal([model], [G]; kwargs...))
end
function loadtotal(models::AbstractVector{<:AbstractModel}, Gs; datapath = "data/simulation", allownegative = false)
    output = Result[]
    
    for (k, model) ∈ enumerate(models)
        G = Gs[k]

        folder = SIMPATHS[typeof(model)]
        controltype = ifelse(allownegative, "allownegative", "nonnegative")
        cachefolder = joinpath(datapath, folder, controltype, "cache")
        filename = makefilename(model, G)
        savepath = joinpath(cachefolder, filename)

        cachefile = jldopen(savepath, "r")

        timekeys = keys(cachefile)
        timesteps = round.(parse.(Float64, timekeys), digits = 4)

        ix = sortperm(timesteps)
        timesteps = timesteps[ix]
        timekeys = timekeys[ix]

        T = length(timesteps)
        F = Array{Float64, 3}(undef, size(G, 1), size(G, 2), T)
        policy = Array{Policy, 3}(undef, size(G, 1), size(G, 2), T)

        for (k, key) ∈ enumerate(timekeys)
            F[:, :, k] .= cachefile[key]["F"]
            policy[:, :, k] .= cachefile[key]["policy"]
        end

        push!(output, (timesteps, F, policy))

        close(cachefile)
    end

    return output
end

function getbool(env, key, fallback)
    key ∈ keys(env) ? env[key] == "true" : fallback
end
function getnumber(env, key, fallback; type = Float64)
    if key ∈ keys(env)
        tol = tryparse(type, env[key])

        if !isnothing(tol)
            return tol
        end
    end

    return fallback
end