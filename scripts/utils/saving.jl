function simpaths(model::AbstractModel)

    basedir = if model isa TippingModel
        "tipping"
    elseif model isa LinearModel
        "linear"
    elseif model isa JumpModel
        "jump"
    else
        throw("Directory not specified for model $(typeof(model))")
    end

    damagedir = if model.damages isa Kalkuhl
        "growth"
    elseif model.damages isa WeitzmanLevel 
        "level"
    else
        throw("Directory not specified for damages $(typeof(model.damages))")
    end

    preferencedir = if model.preferences isa EpsteinZin
        "epsteinzin"
    elseif model.preferences isa CRRA
        "crra"
    elseif model.preferences isa LogSeparable
        "logseparable"
    elseif model.preferences isa LogUtility
        "logutility"
    else
        throw("Directory not specified for preferences $(typeof(model.preferences))")
    end


    return joinpath(basedir, damagedir, preferencedir)
end
function makefilename(model::AbstractModel)
    # Get model-specific parameters
    modelparameters = if model isa TippingModel
        @unpack Tᶜ = model.feedback
        Printf.Format("Tc=%.2f_") => (Tᶜ,)
    else
        Printf.Format("") => ()
    end
    
    # Get preference parameters
    preferenceparameters = if model.preferences isa EpsteinZin
        @unpack ρ, θ, ψ = model.preferences
        Printf.Format("ρ=%.5f_θ=%.2f_ψ=%.2f_") => (ρ, θ, ψ)
    elseif model.preferences isa LogSeparable
        @unpack ρ, θ = model.preferences
        ψ = one(ρ)
        Printf.Format("ρ=%.5f_θ=%.2f_ψ=%.2f_") => (ρ, θ, ψ)
    elseif model.preferences isa CRRA
        @unpack ρ, θ = model.preferences
        Printf.Format("ρ=%.5f_θ=%.2f_") => (ρ, θ)
    elseif model.preferences isa LogUtility
        @unpack ρ = model.preferences
        Printf.Format("ρ=%.5f_") => (ρ,)
    else
        throw("Filename not implemented for preferences $(typeof(model.preferences))")
    end
    
    # Get economy parameters
    @unpack ωᵣ = model.economy
    @unpack σₜ, σₘ = model.hogg
    economyparameters = Printf.Format("σT=%.4f_σm=%.4f_ωr=%.5f") => (σₜ, σₘ, ωᵣ)
    
    # Get damage parameters
    damageparameters = if model.damages isa WeitzmanLevel
        @unpack ξ = model.damages
        Printf.Format("_ξ=%.6f") => (ξ,)
    elseif model.damages isa Kalkuhl
        @unpack ξ₁, ξ₂ = model.damages
        Printf.Format("_ξ1=%.6f_ξ2=%.6f") => (ξ₁, ξ₂)
    else
        Printf.Format("") => ()
    end
    
    # Build filename string
    filenameparameters = String[]
    
    # Add model-specific part
    if !isempty(modelparameters.first.str)
        push!(filenameparameters, Printf.format(modelparameters.first, modelparameters.second...))
    end
    
    # Add preference part
    push!(filenameparameters, Printf.format(preferenceparameters.first, preferenceparameters.second...))
    
    # Add economy part
    push!(filenameparameters, Printf.format(economyparameters.first, economyparameters.second...))
    
    # Add damage part
    if !isempty(string(damageparameters.first))
        push!(filenameparameters, Printf.format(damageparameters.first, damageparameters.second...))
    end
    
    # Join and clean up
    filename = join(filenameparameters, "")
    filename = rstrip(filename, '_')  # Remove trailing underscore
    
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

function loadterminal(model::AbstractModel{T}; outdir = "data/simulation", addpath = "") where T
    folder = simpaths(model)
    filename = makefilename(model)
    
    savedir = joinpath(outdir, folder, "terminal", addpath)
    savepath = joinpath(savedir, filename)

    if !isfile(savepath)
        error("File $filename does not exist in $savedir.\nAvailable files are:\n$(join(readdir(savedir), "\n"))")
    end

    state = load(savepath, "state")
    G = load(savepath, "G")

    return state, G
end
function loadterminal(models::Vector{<:AbstractModel}; outdir = "data/simulation", addpaths = repeat([""], length(models)))
    return [loadterminal(model; outdir = outdir, addpath = addpaths[i]) for (i, model) ∈ enumerate(models)] 
end

function loadtotal(model::AbstractModel{T}; outdir = "data/simulation", loadkwargs...) where T
    
    if !isdir(outdir)
        error("Output directory does not exist: $(outdir)\nHave you not solved the constrained or negative problem?")
    end

    folder = simpaths(model)
    cachefolder = joinpath(outdir, folder)
    
    if !isdir(cachefolder)
        error("Cache folder does not exist: $(folder)\nHave you solved this combination of problems?")
    end
    
    filename = makefilename(model)
    cachepath = joinpath(cachefolder, filename)
    
    if !isfile(cachepath)
        error("Cache file does not exist: $filename\nHave you solved the problem for these parameters?")
    end

    return loadtotal(cachepath; loadkwargs...)
end
function loadtotal(cachepath::String; tspan = (0, Inf))
    cachefile = jldopen(cachepath, "r")
    G = cachefile["G"]
    model = cachefile["model"]
    timekeys = filter(key -> key ∉ ["G", "model"], keys(cachefile))
    timesteps = round.(parse.(Float64, timekeys), digits = 4)

    ix = sortperm(timesteps)
    timesteps = timesteps[ix]
    timekeys = timekeys[ix]

    t₀, t₁ = tspan
    selectidx = t₀ .≤ timesteps .≤ t₁
    timesteps = timesteps[selectidx]
    timekeys = timekeys[selectidx]

    states = DPState[]
    for key ∈ timekeys
        push!(states, cachefile[key]["state"])
    end
    close(cachefile)

    outdict = Dict(timesteps .=> states)

    return outdict, G, model
end

function loadgame(models::Vector{<:AbstractModel}; outdir = "data/simulation")
    filename = makefilename(models)
    cachepath = joinpath(outdir, filename)

    return loadgame(cachepath)
end

function loadgame(cachepath::String)
    cachefile = jldopen(cachepath, "r")
    G = cachefile["G"]
    models = cachefile["models"]
    timekeys = filter(key -> key ∉ ["G", "models"], keys(cachefile))
    timesteps = round.(parse.(Float64, timekeys), digits = 4)

    ix = sortperm(timesteps)
    timesteps = timesteps[ix]
    timekeys = timekeys[ix]

    T = length(timesteps)

    F = Dict{AbstractModel, Array{Float64, 3}}(model => Array{Float64, 3}(undef, size(G)..., T) for model ∈ models)
    policy = Dict{AbstractModel, Array{Float64, 4}}(model => Array{Float64, 4}(undef, size(G)..., 2, T) for model ∈ models)

    for (k, key) in enumerate(timekeys)
        # TODO: This assumes the order of Fs to be the same of models, think of a check for this
        Fs = cachefile[key]["Fs"]
        policies = cachefile[key]["policies"]

        for (j, model) in enumerate(models)
            F[model][:, :, k] .= Fs[j]
            policy[model][:, :, :, k] .= policies[j]
        end
    end

    close(cachefile)

    return timesteps, F, policy, G, models
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