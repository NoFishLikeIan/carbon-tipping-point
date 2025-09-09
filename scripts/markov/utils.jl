mutable struct Error{S}
    absolute::S
    relative::S
end

function Base.isless(error::Error{S}, tolerance::Error{S}) where S
    (error.absolute < tolerance.absolute) && (error.relative < tolerance.relative)
end

function abserror(a::AbstractArray{S}, b::AbstractArray{S}) where S
    abserror!(Error{S}(zero(S), zero(S)), a, b)
end
function abserror!(error::Error{S}, a::AbstractArray{S}, b::AbstractArray{S}) where S
    for k in eachindex(a)
        Δ = abs(a[k] - b[k])
        if Δ > error.absolute
            error.absolute = Δ
            error.relative = Δ / abs(b[k])
        end
    end

    return error
end

function initcachefile(model, G, outdir, withnegative; overwrite = false)
    # Initialise cache folder
    folder = makesimulationpaths(model, withnegative)
    cachefolder = joinpath(outdir, folder)
    
    if !isdir(cachefolder)
        mkpath(cachefolder)
    end

    filename = makefilename(model)
    cachepath = joinpath(cachefolder, filename)

    if isfile(cachepath) && overwrite
        @warn "File $cachepath already exists and mode is overwrite. Will remove."

        rm(cachepath)

        cachefile = jldopen(cachepath, "w+")
        cachefile["G"] = G
        cachefile["model"] = model

    elseif isfile(cachepath) && !overwrite 

        println("File $cachepath already exists and mode is not overwrite. Will resume from cache.")
        cachefile = jldopen(cachepath, "a+")

    else
        cachefile = jldopen(cachepath, "w+")
        cachefile["G"] = G
        cachefile["model"] = model
    end

    return cachepath, cachefile
end