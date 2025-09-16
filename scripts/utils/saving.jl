function makesimulationpaths(model::IAM{S, D, P, C}, withnegative::Bool) where {S, D <: Damages{S}, P <: LogSeparable{S}, C <: Climate{S}}
    basedir = if C <: TippingClimate
        "tipping"
    elseif C <: LinearClimate
        "linear"
    elseif C <: JumpingClimate
        "jump"
    else
        throw("Directory not specified for model $(typeof(model))")
    end

    damagedir = if D <: GrowthDamages
        "growth"
    else
        throw("Directory not specified for damages $(typeof(model.damages))")
    end

    preferencedir = if P <: EpsteinZin
        "epsteinzin"
    elseif P <: CRRA
        "crra"
    elseif P <: LogSeparable
        "logseparable"
    elseif P <: LogUtility
        "logutility"
    else
        throw("Directory not specified for preferences $(typeof(model.preferences))")
    end

    controldir = withnegative ? "negative" : "constrained"

    return joinpath(basedir, damagedir, preferencedir, controldir)
end
function makefilename(model::IAM{S, D, P, C}) where {S, D <: Damages{S}, P <: LogSeparable{S}, C <: Climate{S}}
    # 1. Threshold temperature deviation from pre-industrial
    thresholdpart = if C <: TippingClimate
        deviation = model.climate.feedback.Tᶜ
        "T$(Printf.format(Printf.Format("%.1f"), deviation))"
    else
        "Linear"
    end
    
    damagepart = if D <: Kalkuhl
        "kalkuhl"
    elseif D <: WeitzmanGrowth
        "weitzman"
    elseif D <: NoDamageGrowth
        "no-damages"
    else
        throw("Directory not specified for damages $(typeof(model.damages))")
    end
    
    # 3. RRA θ
    θ = model.preferences.θ
    rrapart = "RRA$(Printf.format(Printf.Format("%.1f"), θ))"
    
    filename = "$(thresholdpart)_$(damagepart)_$(rrapart).jld2"
    return replace(filename, "." => ",")
end

function loadterminal(model::IAM; outdir = "data/simulation", addpath = "")
    folder = makesimulationpaths(model, withnegative)
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

function loadtotal(model::IAM; outdir = "data/simulation", loadkwargs...)
    if !isdir(outdir)
        error("Output directory does not exist: $(outdir)\nHave you not solved the constrained or negative problem?")
    end

    folder = makesimulationpaths(model, withnegative)
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