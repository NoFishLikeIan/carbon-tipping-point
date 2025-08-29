function simpaths(model::AbstractModel, withnegative::Bool)
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
    elseif model.damages isa NoDamageGrowth
        "no-damages"
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

    controldir = withnegative ? "negative" : "constrained"

    return joinpath(basedir, damagedir, preferencedir, controldir)
end
function makefilename(model::AbstractModel)
    # 1. Threshold temperature deviation from pre-industrial
    thresholdpart = if model isa TippingModel
        deviation = model.feedback.Tᶜ - model.hogg.Tᵖ
        "T$(Printf.format(Printf.Format("%.1f"), deviation))"
    else
        "Linear"
    end
    
    # 2. Type of damages
    damagepart = if model.damages isa Kalkuhl
        "Kalkuhl"
    elseif model.damages isa WeitzmanLevel 
        "Weitzman"
    elseif model.damages isa NoDamageGrowth
        "NoDamage"
    else
        "$(typeof(model.damages).name.name)"
    end
    
    # 3. RRA θ
    θ = model.preferences.θ
    rrapart = "RRA$(Printf.format(Printf.Format("%.1f"), θ))"
    
    filename = "$(thresholdpart)_$(damagepart)_$(rrapart).jld2"
    return replace(filename, "." => ",")
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
    folder = simpaths(model, withnegative)
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

    folder = simpaths(model, withnegative)
    cachefolder = joinpath(outdir, folder)
    
    if !isdir(cachefolder)
        error("Cache folder does not exist: $(folder)\nHave you solved this combination of problems?")
    end
    
    filename = makefilename(model)
    filepath = joinpath(cachefolder, filename)
    
    if !isfile(filepath)
        error("Cache file does not exist: $filename\nHave you solved the problem for these parameters?")
    end

    return loadtotal(filepath; loadkwargs...)
end
function loadtotal(filepath::String; tspan = (0, Inf))
    cachefile = jldopen(filepath, "r")
    G = cachefile["G"]
    model = cachefile["model"]
    timekeys = filter(key -> key ∉ ("G", "model"), keys(cachefile))
    timesteps = round.(parse.(Float64, timekeys), digits = 4)

    ix = sortperm(timesteps)
    timesteps = timesteps[ix]
    timekeys = timekeys[ix]

    t₀, t₁ = tspan
    selectidx = t₀ .≤ timesteps .≤ t₁
    timesteps = timesteps[selectidx]
    timekeys = timekeys[selectidx]

    N₁, N₂ = size(G)
    S = eltype(G)
    values = ValueFunction{S, N₁, N₂}[]
    for key ∈ timekeys
        V = cachefile[key]["V"]
        push!(values, V)
    end
    
    close(cachefile)

    outdict = OrderedDict(timesteps .=> values)

    return outdict, model, G
end

function loadproblem(filepath)
    cachefile = jldopen(filepath, "r")
    G = cachefile["G"]
    model = cachefile["model"]

    close(cachefile)

    return model, G
end

function loadgame(models::Vector{<:AbstractModel}; outdir = "data/simulation")
    filename = makefilename(models)
    filepath = joinpath(outdir, filename)

    return loadgame(filepath)
end

function loadgame(filepath::String)
    cachefile = jldopen(filepath, "r")
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
    for (root, _, filenames) in walkdir(simpath)
        if any(occursin.(exclude, root))
            continue
        end

        for filename in filenames
            if occursin("jld2", filename)
                push!(files, joinpath(root, filename))
            end
        end
    end

    return files
end