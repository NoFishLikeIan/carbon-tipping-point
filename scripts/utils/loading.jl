"Load a single solved policy file and build its abatement interpolation."
function loadpolicy(filepath::String; tspan = (0., Inf))
	values, _, G = loadtotal(filepath; tspan)
	_, αitp = buildinterpolations(values, G)

	return αitp
end

"Load all solved policy files found under `simpath` and return their interpolations and metadata."
function loadallpolicies(simpath::String; tspan = (0., Inf), exclude = ["terminal"])
	files = listfiles(simpath; exclude)

	policies = OrderedDict{String, Interpolations.Extrapolation}()
	for filepath in files
		policies[filepath] = loadpolicy(filepath; tspan)
	end

	return policies
end

"Load policies in `paths` into an array"
function loadallpoliciesarrays(paths::PS; tspan = (0., Inf)) where {S, PS <: OrderedDict{S, String}}
    firstpath = paths.vals[1]
    _, G = loadproblem(firstpath)

    return loadallpoliciesarrays(paths, G; tspan)
end
function loadallpoliciesarrays(paths::PS, G::GR; tspan = (0., Inf)) where {S, N₁, N₂, PS <: OrderedDict{S, String}, GR <: RegularGrid{N₁, N₂, S}}
    thresholds = paths.keys
    
    

end

"Parse critical threshold `Tᶜ` from a policy filename, such that, `T2,00_burke_RRA10,00.jld2 → 2.0`"
function parsethreshold(filepath::String)
    filename, _ = splitext(basename(filepath))
    parts = split(filename, "_")
    thresholdpart = first(parts)

    if thresholdpart == "Linear"
        return Inf
    end

    thresholdstr = replace(thresholdpart, "," => ".", "T" => "")
    
    return parse(Float64, thresholdstr)
end

"Return `OrderedDict` with `threshold => filepath` mapping for solved threshold policies in `simpath`."
function loadsimulationpaths(simpath::String; exclude = ["terminal"])
    files = listfiles(simpath; exclude)
    
    indexed = OrderedDict{Float64, String}()
    for filepath in files
        Tᶜ = parsethreshold(filepath)
        indexed[Tᶜ] = filepath
    end

    sort!(indexed)
    
    return indexed
end

"Construct policy matrix `A` from `paths`, applies a permutation QR-factorisation and returns the permutation indices `k`"
function basispolicypaths(paths::OrderedDict{Float64, String}, K::Int; tspan = (0, Inf), kwargs...)
    firstpath = values(paths)[1]
    _, G = loadproblem(firstpath; tspan)

    return basispolicypaths(paths, K, G; tspan, kwargs...)
end
function basispolicypaths(paths::OrderedDict{Float64, String}, K::Int, G::RegularGrid; tspan = (0, Inf))
    thresholds = paths.keys
    firstpath = first(paths.vals)
    v = first(loadtotal(firstpath; tspan))
    ts = v.keys

    n = length(thresholds)
    m = length(G) * length(ts)
    S = eltype(G)
    A = Matrix{S}(undef, m, n)

    for (k, path) in enumerate(paths.vals)
        vₖ, _, Gₖ = loadtotal(path; tspan)
        projvₖ = OrderedDict(t => interpolateovergrid(vₖₜ, Gₖ, G) for (t, vₖₜ) in vₖ)

        _, αitp = buildinterpolations(projvₖ, G)

        A[:, k] = vec(αitp.itp.coefs)
    end

    res = qr!(A, ColumnNorm())
    kthresholds = thresholds[res.p][1:K]

    kpaths = OrderedDict(Tᶜ => path for (Tᶜ, path) in paths if Tᶜ ∈ kthresholds)
    sort!(kpaths)
    
    return kpaths
end