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
    JumpModel{GrowthDamages, EpsteinZin}  => "jump/growth"
)

function makefilename(model::TippingModel{LevelDamages, EpsteinZin})
    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ, σₘ = model.hogg
    @unpack Tᶜ = model.albedo
    @unpack ξ = model.damages

    filename = @sprintf("Tc=%.2f_ρ=%.5f_θ=%.2f_ψ=%.2f_σT=%.4f_σm=%.4f_ωr=%.5f_ξ=%.6f", Tᶜ, ρ, θ, ψ, σₜ, σₘ, ωᵣ, ξ)

    return "$(replace(filename, "." => ",")).jld2"
end

function makefilename(model::TippingModel{GrowthDamages, EpsteinZin})
    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ, σₘ = model.hogg
    @unpack Tᶜ = model.albedo
    @unpack ξ, υ = model.damages

    filename = @sprintf("Tc=%.2f_ρ=%.5f_θ=%.2f_ψ=%.2f_σT=%.4f_σm=%.4f_ωr=%.5f_ξ=%.6f_υ=%.3f", Tᶜ, ρ, θ, ψ, σₜ, σₘ, ωᵣ, ξ, υ)

    return "$(replace(filename, "." => ",")).jld2"
end

function makefilename(model::JumpModel{GrowthDamages, EpsteinZin})
    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ, σₘ = model.hogg
    @unpack ξ, υ = model.damages

    filename = @sprintf("ρ=%.5f_θ=%.2f_ψ=%.2f_σT=%.4f_σm=%.4f_ωr=%.5f_ξ=%.6f_υ=%.3f", ρ, θ, ψ, σₜ, σₘ, ωᵣ, ξ, υ)

    return "$(replace(filename, "." => ",")).jld2"
end

function makefilename(model::JumpModel{LevelDamages, EpsteinZin})
    @unpack ρ, θ, ψ = model.preferences
    @unpack ωᵣ = model.economy
    @unpack σₜ, σₘ = model.hogg
    @unpack ξ = model.damages

    filename = @sprintf("ρ=%.5f_θ=%.2f_ψ=%.2f_σT=%.4f_σm=%.4f_ωr=%.5f_ξ=%.6f", ρ, θ, ψ, σₜ, σₘ, ωᵣ, ξ)

    return "$(replace(filename, "." => ",")).jld2"
end

function makefilename(models::Vector{<:AbstractModel})

    filenames = String[]

    for model in models
        filename = makefilename(model)

        push!(filenames, replace(filename, ".jld2" => ""))
    end

    return "$(join(filenames, "-")).jld2"
end

function loadterminal(model::AbstractModel; outdir = "data/simulation", addpath = "")
    folder = SIMPATHS[typeof(model)]
    filename = makefilename(model)
    
    savedir = joinpath(outdir, folder, "terminal", addpath)
    savepath = joinpath(savedir, filename)

    if !isfile(savepath)
        error("File $filename does not exist in $savedir.\nAvailable files are:\n$(join(readdir(savedir), "\n"))")
    end

    F̄ = load(savepath, "F̄")
    policy = load(savepath, "policy")
    G = load(savepath, "G")

    return F̄, policy, G
end
function loadterminal(models::Vector{<:AbstractModel}; outdir = "data/simulation", addpaths = repeat([""], length(models)))
    return [loadterminal(model; outdir = outdir, addpath = addpaths[i]) for (i, model) ∈ enumerate(models)] 
end


function loadtotal(model::AbstractModel; outdir = "data/simulation")
    folder = SIMPATHS[typeof(model)]
    cachefolder = joinpath(outdir, folder)
    filename = makefilename(model)
    cachepath = joinpath(cachefolder, filename)

    return loadtotal(cachepath)
end

function loadtotal(cachepath::String)
    cachefile = jldopen(cachepath, "r")
    G = cachefile["G"]
    model = cachefile["model"]
    timekeys = filter(key -> key ∉ ["G", "model"], keys(cachefile))
    timesteps = round.(parse.(Float64, timekeys), digits = 4)

    ix = sortperm(timesteps)
    timesteps = timesteps[ix]
    timekeys = timekeys[ix]

    T = length(timesteps)
    F = Array{Float64, 3}(undef, size(G)..., T)
    policy = Array{Float64, 4}(undef, size(G)..., 2, T)

    for (k, key) ∈ enumerate(timekeys)
        F[:, :, k] .= cachefile[key]["F"]
        policy[:, :, :, k] .= cachefile[key]["policy"]
    end

    close(cachefile)

    return timesteps, F, policy, G, model
end

function listfiles(simpath::String; exclude = ["terminal"])
    if !isdir(simpath)
        error("Directory does not exist: $simpath")
    end

    files = String[]
    for (root, _, file_names) in walkdir(simpath)
        if any(occursin.(exclude, root))
            continue
        end

        for file_name in file_names
            push!(files, joinpath(root, file_name))
        end
    end

    return files
end