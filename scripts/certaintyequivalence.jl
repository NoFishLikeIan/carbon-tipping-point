using Pkg
Pkg.resolve(); Pkg.instantiate();

using Base.Threads: nthreads
using UnPack: @unpack
using Dates: now

include("arguments.jl") # Import argument parser

parsedargs = ArgParse.parse_args(ceargstable)

@unpack simulationpath, eis, rra, remotethreshold, verbose, datapath = parsedargs

if (verbose ≥ 1)
    println("$(now()): ", "Running with $(nthreads()) threads for ψ=$eis, θ=$rra and remote threshold=$remotethreshold...")
    flush(stdout)
end

# Begin script
using Model, Grid

using Interpolations
using Roots
using QuadGK

include("utils/saving.jl")
include("markov/forward.jl")

calibrationdirectory = joinpath(datapath, "calibration.jld2")
calibration = load_object(calibrationdirectory)

damages = Kalkuhl()
hogg = Hogg()
preferences = EpsteinZin(θ = rra, ψ = eis);
economy = Economy()

imminentmodel = TippingModel(Albedo(1.5), hogg, preferences, damages, economy)
remotemodel = TippingModel(Albedo(remotethreshold), hogg, preferences, damages, economy)

# Constructs wishful thinker and prudent policies
outdir = joinpath(datapath, simulationpath, "constrained")

timestepsimm, _, imminentpolicy, _ = loadtotal(imminentmodel; outdir)
timestepsrem, _, remotepolicy, G = loadtotal(remotemodel; outdir)

# Ensure the same size of the two policies FIXME: This is very ugly
if length(timestepsrem) < length(timestepsimm)
    extendedremote = similar(imminentpolicy)
    indxs = findall(in(timestepsrem), timestepsimm)
    extendedremote[:, :, :, indxs] .= remotepolicy
    
    outdxs = findall(!in(timestepsrem), timestepsimm)

    for idx in outdxs
        _, closest = findmin(jdx -> abs(idx - jdx), indxs)
        extendedremote[:, :, :, idx] .= remotepolicy[:, :, :, indxs[closest]]
    end

    remotepolicy = extendedremote

elseif length(timestepsrem) > length(timestepsimm)
    extendedimminent = similar(remotepolicy)
    indxs = findall(in(timestepsimm), timestepsrem)
    extendedimminent[:, :, :, indxs] .= imminentpolicy
    
    outdxs = findall(!in(timestepsimm), timestepsrem)

    for idx in outdxs
        _, closest = findmin(jdx -> abs(idx - jdx), indxs)
        extendedimminent[:, :, :, idx] .= imminentpolicy[:, :, :, indxs[closest]]
    end

    imminentpolicy = extendedimminent
end

@assert size(imminentpolicy) == size(remotepolicy)

wfpolicy = NaN * zeros(size(remotepolicy))

imminenttipregion = @. getindex(G, 1) ≥ 1.5 + hogg.Tᵖ
wfpolicy[imminenttipregion, :, :] .= imminentpolicy[imminenttipregion, :, :]
wfpolicy[.~imminenttipregion, :, :] .= remotepolicy[.~imminenttipregion, :, :]

prudpolicy = NaN * zeros(size(remotepolicy))

remotetippedregion = @. getindex(G, 1) ≥ remotethreshold + hogg.Tᵖ
prudpolicy[remotetippedregion, :, :] .= remotepolicy[remotetippedregion, :, :]
prudpolicy[.~remotetippedregion, :, :] .= imminentpolicy[.~remotetippedregion, :, :]

Tspace = range(G.domains[1]...; length = size(G, 1))
mspace = range(G.domains[2]...; length = size(G, 2))

nodes = (Tspace, mspace, timestepsimm)

χwfitp = linear_interpolation(nodes, wfpolicy[:, :, 1, :]; extrapolation_bc = Line())
αwfitp = linear_interpolation(nodes, wfpolicy[:, :, 2, :]; extrapolation_bc = Line())

χpitp = linear_interpolation(nodes, prudpolicy[:, :, 1, :]; extrapolation_bc = Line())
αpitp = linear_interpolation(nodes, prudpolicy[:, :, 2, :]; extrapolation_bc = Line())

χremoteitp = linear_interpolation(nodes, remotepolicy[:, :, 1, :]; extrapolation_bc = Line())
αremoteitp = linear_interpolation(nodes, remotepolicy[:, :, 2, :]; extrapolation_bc = Line())

χimminentitp = linear_interpolation(nodes, imminentpolicy[:, :, 1, :]; extrapolation_bc = Line())
αimminentitp = linear_interpolation(nodes, imminentpolicy[:, :, 2, :]; extrapolation_bc = Line())

# Compute the climate change cost the wishful thinker and the prudent policies
if (verbose ≥ 1)
    println("$(now()): ","Running forward mode...")
    flush(stdout)
end

(verbose ≥ 1) && println("$(now()): ","Remote optimal...")
F̄ = computebackward(χremoteitp, αremoteitp, remotemodel, calibration; outdir, verbose)

(verbose ≥ 1) && println("$(now()): ","Imminent optimal...")
F̲ = computebackward(χimminentitp, αimminentitp, remotemodel, calibration; outdir, verbose)


(verbose ≥ 1) && println("$(now()): ","Wishful thinker...")
Fw = computebackward(χwfitp, αwfitp, imminentmodel, calibration; outdir, verbose)


(verbose ≥ 1) && println("$(now()): ","Cautious...")
Fp = computebackward(χpitp, αpitp, remotemodel, calibration; outdir, verbose)

# Get values at X₀
X₀ = [imminentmodel.hogg.T₀, log(imminentmodel.hogg.M₀)];
Y₀ = imminentmodel.economy.Y₀;

function getV₀(F, X₀, model)
    F₀ = linear_interpolation((Tspace, mspace), F)(X₀[1], X₀[2])
    
    θ = model.preferences.θ
    F₀ * Y₀^(1 - θ) / (1 - θ)
end


# Only care about relative size
V̄ = getV₀(F̄, X₀, imminentmodel)
V̲ = getV₀(F̲, X₀, imminentmodel)
Vw = getV₀(Fw, X₀, imminentmodel)
Vp = getV₀(Fp, X₀, imminentmodel)

# FIXME: It does not find a root for ψ ≥ 1.
# Certainty equilvanece
function ∫f(x, V, model)
    QuadGK.quadgk(
        t -> f(x * exp(economy.ϱ * t), V, model.preferences), 0., 1_000
    ) |> first
end

function ce(V₀, model; interval = (0.005, Inf))
    a = f(interval[1], V₀, model.preferences) - V₀
    b = f(interval[2], V₀, model.preferences) - V₀

    if a * b > 0.
        throw("Interval is not bracketing: ∫f ∈ [$a, $b]")
    end

    find_zero(x ->  ∫f(x, V₀, model) - V₀, interval)
end

if (verbose ≥ 1)
    println("$(now()): ","Computing certainty equivalence...")
    flush(stdout)
end

ceᵖ = ce(Vp, remotemodel)
ceʷ = ce(Vw, imminentmodel) 
cē = ce(V̄, remotemodel) # Optimal with remote
ce̲ = ce(V̲, imminentmodel) # Optimal with imminent

results = Dict(
    "ceᵖ" => ceᵖ,
    "ceʷ" => ceʷ,
    "cē" => cē,
    "ce̲" => ce̲
)

paramname = @sprintf("θ=%.2f_ψ=%.2f_threshold=%.2f", rra, eis, remotethreshold)
filename = "$(replace(paramname, "." => ",")).jld2"

outpath = joinpath(datapath, "certaintyequivalence")
if !isdir(outpath) mkdir(outpath) end

outfile = joinpath(outpath, filename) 
JLD2.save_object(outfile, results)