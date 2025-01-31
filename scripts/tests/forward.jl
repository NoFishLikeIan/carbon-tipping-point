using Revise
using Test: @test
using UnPack: @unpack
using BenchmarkTools
using JLD2
using Interpolations

includet("../utils/saving.jl")
includet("../markov/forward.jl")

begin
    calibration = load_object("data/calibration.jld2")
    damages = GrowthDamages()
    hogg = Hogg()
    preferences = EpsteinZin(ψ = 0.75, θ = 10.)
    economy = Economy()

    imminentmodel = TippingModel(Albedo(1.5), hogg, preferences, damages, economy)
    remotemodel = TippingModel(Albedo(2.5), hogg, preferences, damages, economy)
end;

begin # Constructs wishful thinker and prudent policies
    simpath = "data/simulation-large/constrained"

    F, _, G = loadterminal(imminentmodel; outdir = simpath);

    timestepsimm, _, imminentpolicy, _ = loadtotal(imminentmodel; outdir = simpath)
    timestepsrem, _, remotepolicy, _ = loadtotal(remotemodel; outdir = simpath)

    remotepolicy = remotepolicy[:, :, :, findall(in(timestepsimm), timestepsrem)]; # Assert the same size

    wfpolicy = similar(imminentpolicy);

    tippedregion = @. getindex(G.X, 1) ≥ 1.5 + hogg.Tᵖ
    wfpolicy[tippedregion, :, :] .= imminentpolicy[tippedregion, :, :]
    wfpolicy[.~tippedregion, :, :] .= remotepolicy[.~tippedregion, :, :]

    prudpolicy = similar(imminentpolicy);

    tippedregion = @. getindex(G.X, 1) ≥ 2.5 + hogg.Tᵖ
    prudpolicy[tippedregion, :, :] .= remotepolicy[tippedregion, :, :]
    prudpolicy[.~tippedregion, :, :] .= imminentpolicy[.~tippedregion, :, :]
end;

begin
    Tspace = range(G.domains[1]...; length = size(G, 1))
    mspace = range(G.domains[2]...; length = size(G, 2))

    nodes = (Tspace, mspace, timestepsimm)
    χitp = linear_interpolation(nodes, wfpolicy[:, :, 1, :]; extrapolation_bc = Line())
    αitp = linear_interpolation(nodes, wfpolicy[:, :, 2, :]; extrapolation_bc = Line())
end;

model = imminentmodel
F₀ = computebackward(χitp, αitp, model, G; outdir = simpath, verbose = 2);
