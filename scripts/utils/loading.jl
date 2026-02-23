"Load a single solved policy file and build its abatement interpolation."
function loadpolicy(filepath::String; tspan = (0, Inf))
	values, _, G = loadtotal(filepath; tspan)
	_, αitp = buildinterpolations(values, G)

	return αitp
end

"Load all solved policy files found under `simpath` and return their interpolations and metadata."
function loadallpolicies(simpath::String; tspan = (0, Inf), exclude = ["terminal"])
	files = listfiles(simpath; exclude)

	policies = OrderedDict{String, Interpolations.Extrapolation}()
	for filepath in files
		policies[filepath] = loadpolicy(filepath; tspan)
	end

	return policies
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
function loadregretpolicypaths(simpath::String; exclude = ["terminal"])
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
function filterpolicies(paths::OrderedDict{Float64, String}, K; tspan = (0, Inf))
    firstpath = first(values(paths))
    # Assumes `G` and `ts` are common to all files in `paths`
    vs, _, G = loadtotal(firstpath; tspan)
    ts = keys(vs)

    thresholds = collect(keys(paths))
    n = length(thresholds)
    m = length(G) * length(ts)
    S = eltype(G)
    A = Matrix{S}(undef, m, n)

    for (k, path) in enumerate(values(paths))
        αitp = loadpolicy(path; tspan)
        A[:, k] = vec(αitp.itp.coefs)
    end

    res = qr!(A, ColumnNorm())
    kthresholds = thresholds[res.p][1:K]

    kpaths = OrderedDict(Tᶜ => path for (Tᶜ, path) in paths if Tᶜ ∈ kthresholds)
    sort!(kpaths)
    
    return kpaths
end

struct SimplexPolicies{S, I <: Interpolations.Extrapolation{S}, PS <: OrderedDict{S, I}}
    policies::PS
end

function SimplexPolicies(paths; tspan = (0, Inf))
    policies = OrderedDict(Tᶜ => loadpolicy(path; tspan) for (Tᶜ, path) in paths)

    return SimplexPolicies(policies)
end

