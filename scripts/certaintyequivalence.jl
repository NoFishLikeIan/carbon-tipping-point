using Model, Grid

using Interpolations
using Roots
using QuadGK

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

    wfpolicy = similar(imminentpolicy);

    tippedregion = @. getindex(G.X, 1) ≥ 1.5 + hogg.Tᵖ
    wfpolicy[tippedregion, :, :] .= imminentpolicy[tippedregion, :, :]
    wfpolicy[.~tippedregion, :, :] .= remotepolicy[.~tippedregion, :, :]

    prudpolicy = similar(imminentpolicy);

    tippedregion = @. getindex(G.X, 1) ≥ 2.5 + hogg.Tᵖ
    prudpolicy[tippedregion, :, :] .= remotepolicy[tippedregion, :, :]
    prudpolicy[.~tippedregion, :, :] .= imminentpolicy[.~tippedregion, :, :]

    Tspace = range(G.domains[1]...; length = size(G, 1))
    mspace = range(G.domains[2]...; length = size(G, 2))

    nodes = (Tspace, mspace, timestepsimm)

    χwfitp = linear_interpolation(nodes, wfpolicy[:, :, 1, :]; extrapolation_bc = Line())
    αwfitp = linear_interpolation(nodes, wfpolicy[:, :, 2, :]; extrapolation_bc = Line())

    χpitp = linear_interpolation(nodes, prudpolicy[:, :, 1, :]; extrapolation_bc = Line())
    αpitp = linear_interpolation(nodes, prudpolicy[:, :, 2, :]; extrapolation_bc = Line())
end;

# Compute the climate change cost the wishful thinker and the prudent policies
begin
    Fw = computebackward(simpath, χwfitp, αwfitp, imminentmodel; verbose = 1);
    Fp = computebackward(simpath, χpitp, αpitp, remotemodel; verbose = 1)
end;

# Loads initial optimal values
F̄ = loadtotal(remotemodel; outdir = simpath)[2][:, :, 1];
F̲ = loadtotal(imminentmodel; outdir = simpath)[2][:, :, 1];

# Get values at X₀
X₀ = [imminentmodel.hogg.T₀, log(imminentmodel.hogg.M₀)];
Y₀ = imminentmodel.economy.Y₀;

function getV₀(F, X₀)
    F₀ = linear_interpolation((Tspace, mspace), F)(X₀[1], X₀[2])
    
    θ = 15.
    F₀ * Y₀^(1 - θ) / (1 - θ)
end

V̄ = getV₀(F̄, X₀)
V̲ = getV₀(F̲, X₀)
Vw = getV₀(Fw, X₀)
Vp = getV₀(Fp, X₀)

# Certainty equilvanece
function ce(V₀, model; interval = (0.005, 10_000.))
    a = f(interval[1], V₀, model.preferences) - V₀
    b = f(interval[2], V₀, model.preferences) - V₀

    if a * b > 0.
        throw("Interval is not bracketing: ∫f ∈ [$a, $b]")
    end

    find_zero(x -> f(x, V₀, model.preferences) - V₀, interval)
end

ceʷ = ce(Vw, imminentmodel) # Reckless 
ceᵖ = ce(Vp, remotemodel) # Prudent

cē = ce(V̄, remotemodel) # Optimal with remote
ce̲ = ce(V̲, imminentmodel) # Optimal with imminent

labels = ["ceʷ", "ceᵖ", "cē", "ce̲"];

for (i, ce) in enumerate((ceʷ, ceᵖ, cē, ce̲))
    per = round(100 * ce / Y₀, digits = 2)

    println("$(labels[i]) : $(round(ce, digits = 2)) ($per %)")
end

