using Model
using Grid: Policy, RegularGrid
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

function makefilename(model::TippingModel{LevelDamages, EpsteinZin})
    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ, σₘ = model.hogg
    @unpack Tᶜ = model.albedo
    @unpack ξ = model.damages

    filename = @sprintf("Tc=%.2f_ρ=%.5f_θ=%.2f_ψ=%.2f_σT=%.4f_σm=%.4f_ωᵣ=%.5f_ξ=%.6f", Tᶜ, ρ, θ, ψ, σₜ, σₘ, ωᵣ, ξ)

    return "$(replace(filename, "." => ",")).jld2"
end

function makefilename(model::TippingModel{GrowthDamages, EpsteinZin})
    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ, σₘ = model.hogg
    @unpack Tᶜ = model.albedo
    @unpack ξ, υ = model.damages

    filename = @sprintf("Tc=%.2f_ρ=%.5f_θ=%.2f_ψ=%.2f_σT=%.4f_σm=%.4f_ωᵣ=%.5f_ξ=%.6f_υ=%.3f", Tᶜ, ρ, θ, ψ, σₜ, σₘ, ωᵣ, ξ, υ)

    return "$(replace(filename, "." => ",")).jld2"
end
function makefilename(model::TippingGameModel{GrowthDamages, EpsteinZin})
    @unpack Tᶜ = model.albedo
    ξh, ξl = getproperty.(model.damages, :ξ)
    υh, υl = getproperty.(model.damages, :υ)

    filename = @sprintf("Tc=%.2f_ξh=%.6f_ξl=%.6f_υh=%.3f_υl=%.3f", Tᶜ, ξh, ξl, υh, υl)

    return "$(replace(filename, "." => ",")).jld2"
end

function makefilename(model::JumpModel{GrowthDamages, EpsteinZin})
    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ, σₘ = model.hogg
    @unpack ξ, υ = model.damages

    filename = @sprintf("ρ=%.5f_θ=%.2f_ψ=%.2f_σT=%.4f_σm=%.4f_ωᵣ=%.5f_ξ=%.6f_υ=%.3f", ρ, θ, ψ, σₜ, σₘ, ωᵣ, ξ, υ)

    return "$(replace(filename, "." => ",")).jld2"
end

function makefilename(model::JumpModel{LevelDamages, EpsteinZin})
    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ, σₘ = model.hogg
    @unpack ξ = model.damages

    filename = @sprintf("ρ=%.5f_θ=%.2f_ψ=%.2f_σT=%.4f_σm=%.4f_ωᵣ=%.5f_ξ=%.6f", ρ, θ, ψ, σₜ, σₘ, ωᵣ, ξ)

    return "$(replace(filename, "." => ",")).jld2"
end

function loadterminal(model::AbstractModel; datapath = "data/simulation", addpath = "")
    folder = SIMPATHS[typeof(model)]
    filename = makefilename(model)
    savepath = joinpath(datapath, folder, "terminal", addpath, filename)
    F̄ = load(savepath, "F̄")
    policy = load(savepath, "policy")
    G = load(savepath, "G")

    return F̄, policy, G
end

function loadtotal(model::AbstractGameModel; datapath = "data/simulation", allownegative = false)     
    folder = SIMPATHS[typeof(model)]
    controltype = ifelse(allownegative, "allownegative", "nonnegative")
    cachefolder = joinpath(datapath, folder, controltype)
    filename = makefilename(model)
    savepath = joinpath(cachefolder, filename)

    cachefile = jldopen(savepath, "r")
    G = cachefile["G"]
    timekeys = filter(!=("G"), keys(cachefile))
    timesteps = round.(parse.(Float64, timekeys), digits = 4)

    ix = sortperm(timesteps)
    timesteps = timesteps[ix]
    timekeys = timekeys[ix]

    T = length(timesteps)
    M = size(cachefile[first(timekeys)]["F"], 3)
    F = Array{Float64, 5}(undef, size(G, 1), size(G, 2), M, 2, T)
    policy = Array{Policy, 5}(undef, size(G, 1), size(G, 2), M, 2, T)

    for (k, key) ∈ enumerate(timekeys)
        F[:, :, :, :, k] .= cachefile[key]["F"]
        policy[:, :, :, :, k] .= cachefile[key]["policy"]
    end

    close(cachefile)

    return timesteps, F, policy, G
end

function loadtotal(model::AbstractPlannerModel; datapath = "data/simulation", allownegative = false)
    folder = SIMPATHS[typeof(model)]
    controltype = ifelse(allownegative, "allownegative", "nonnegative")
    cachefolder = joinpath(datapath, folder, controltype)
    filename = makefilename(model)
    savepath = joinpath(cachefolder, filename)

    cachefile = jldopen(savepath, "r")
    G = cachefile["G"]
    timekeys = filter(!=("G"), keys(cachefile))
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

    close(cachefile)

    return timesteps, F, policy, G
end

function getbool(env, key, fallback)
    key ∈ keys(env) ? env[key] == "true" : fallback
end
function getnumber(env, key, fallback; type = Float64)
    if key ∈ keys(env)
        v = env[key]
        tol = parse(type, v)
        return tol
    end

    return fallback
end