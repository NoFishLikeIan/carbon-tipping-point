using Model, Grid

using Interpolations
using Roots
using QuadGK
using Plots

include("utils/saving.jl")
include("markov/forward.jl")

begin
    calibration = load_object("data/calibration.jld2")
    damages = GrowthDamages()
    hogg = Hogg()
    preferences = EpsteinZin(ψ = 0.75, θ = 10.)
    economy = Economy()

    imminentmodel = TippingModel(Albedo(1.5), hogg, preferences, damages, economy, calibration)
    remotemodel = TippingModel(Albedo(2.5), hogg, preferences, damages, economy, calibration)
end;

begin # Constructs wishful thinker and prudent policies
    simpath = "data/simulation-large/constrained"

    timestepsimm, _, imminentpolicy, _ = loadtotal(imminentmodel; outdir = simpath)
    timestepsrem, _, remotepolicy, G = loadtotal(remotemodel; outdir = simpath)

    remotepolicy = remotepolicy[:, :, :, findall(in(timestepsimm), timestepsrem)]; # Assert the same size

    wfpolicy = NaN * zeros(size(remotepolicy))

    imminenttipregion = @. getindex(G.X, 1) ≥ 1.5 + hogg.Tᵖ
    wfpolicy[imminenttipregion, :, :] .= imminentpolicy[imminenttipregion, :, :]
    wfpolicy[.~imminenttipregion, :, :] .= remotepolicy[.~imminenttipregion, :, :]

    prudpolicy = NaN * zeros(size(remotepolicy))

    remotetippedregion = @. getindex(G.X, 1) ≥ 2.5 + hogg.Tᵖ
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
end;

Δwf = (wfpolicy[:, :, 2, 1] .- prudpolicy[:, :, 2, 1])
heatmap(mspace, Tspace, Δwf; clims = maximum(abs.(Δwf)) .* (-1, 1), c = :coolwarm)

# Compute the climate change cost the wishful thinker and the prudent policies


begin
    F̄ = computebackward(simpath, χremoteitp, αremoteitp, remotemodel; verbose = 1)
    F̲ = computebackward(simpath, χimminentitp, αimminentitp, remotemodel; verbose = 1)
    Fw = computebackward(simpath, χwfitp, αwfitp, imminentmodel; verbose = 1)
    Fp = computebackward(simpath, χpitp, αpitp, remotemodel; verbose = 1)
end;

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

ceᵖ = ce(Vp, remotemodel)
ceʷ = ce(Vw, imminentmodel) 
cē = ce(V̄, remotemodel) # Optimal with remote
ce̲ = ce(V̲, imminentmodel) # Optimal with imminent

labels = ["ceʷ", "ceᵖ", "cē", "ce̲"];

function printce(ce, label; digits = 3)
    per = round(100 * ce / Y₀, digits = digits)

    println("$(label) : $(round(ce, digits = digits)) / ($per %)")
end

for (i, ce) in enumerate((ceʷ, ceᵖ, cē, ce̲))
    printce(ce, labels[i])
end

printce(ce̲ - ceʷ, "ce̲ - ceʷ")
printce(cē - ceᵖ, "cē - ceᵖ")
printce(cē - ce̲, "cē - ce̲")

printce((ce̲ - ceʷ) - (cē - ceᵖ), "(cē - ceᵖ) - (ce̲ - ceʷ)")
