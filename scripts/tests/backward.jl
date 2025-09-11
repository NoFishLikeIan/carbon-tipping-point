using Test, BenchmarkTools, Revise, UnPack
using Plots, LaTeXStrings
default(c=:viridis, label=false, dpi=180)

using Model, Grid
using Base.Threads
using SciMLBase
using Statistics
using StaticArrays, SparseArrays
using Interpolations

using LinearSolve, LinearAlgebra

using JLD2, UnPack
using Dates, Printf

includet("../../src/valuefunction.jl")
includet("../../src/extend/model.jl")
includet("../../src/extend/grid.jl")
includet("../../src/extend/valuefunction.jl")
includet("../utils/saving.jl")
includet("../markov/utils.jl")
includet("../markov/chain.jl")
includet("../markov/finitedifference.jl")

begin # Construct the model
    DATAPATH = "data"
    calibrationpath = joinpath(DATAPATH, "calibration")

    # Load economic calibration
    abatementpath = joinpath(calibrationpath, "abatement.jld2")
    @assert isfile(abatementpath) "Abatement calibration file not found at $abatementpath"
    abatementfile = jldopen(abatementpath, "r+")
    @unpack abatement = abatementfile
    close(abatementfile)

    investments = Investment()
    damages = Kalkuhl()
    economy = Economy(investments = investments, damages = damages, abatement = abatement)

    # Load climate claibration
    climatepath = joinpath(calibrationpath, "climate.jld2")
    @assert isfile(climatepath) "Climate calibration file not found at $climatepath"
    climatefile = jldopen(climatepath, "r+")
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher, decay = climatefile
    close(climatefile)

    threshold = 2.5
    climate = if 0 < threshold < Inf
        feedback = Model.updateTᶜ(threshold + hogg.Tᵖ, feedback)
        TippingClimate(hogg, decay, feedback)
    else
        LinearClimate(hogg, decay)
    end

    preferences = LogSeparable()

    model = IAM(climate, economy, preferences)
    deterministicmodel = determinsticIAM(model)
end

begin
    N₁ = 200; N₂ = 100;
    N = (N₁, N₂)
    Tdomain = hogg.Tᵖ .+ (0.5, 7.)
    mmin = mstable(Tdomain[1] + 0.5, model.climate)
    mmax = mstable(Tdomain[2] - 0.5, model.climate)
    mdomain = (mmin, mmax)
    domains = (Tdomain, mdomain)
    
    Gterminal = constructelasticgrid(N, domains, model)
    Δt = 1 / 200
    τ = 500.
    withnegative = false
end;

terminalvaluefunction = ValueFunction(τ, climate, Gterminal, calibration)
steadystate!(terminalvaluefunction, Δt, deterministicmodel, Gterminal, calibration; verbose = 1, tolerance = Error(1e-3, 1e-3), timeiterations = 1500, withnegative = withnegative, picarditerations = 0, printstep = 50)

begin # HJB error
    Ā = constructA!(terminalvaluefunction, 1 / Δt, model, Gterminal, calibration, withnegative)
    b̄ = constructb(terminalvaluefunction, 1 / Δt, model, Gterminal, calibration)

    hjb = reshape(Ā * vec(terminalvaluefunction.H) - b̄, size(Gterminal))
end;

if isinteractive()
    Tspace, mspace = Gterminal.ranges
    hjberrorfig = heatmap(mspace, Tspace, hjb; title = "HJB error", xlabel = L"m", ylabel = L"T", c=:Greens, xlims = extrema(mspace), ylims = extrema(Tspace))
end

if isinteractive()
    E = ε(terminalvaluefunction, model, calibration, Gterminal)
    nullcline = [mstable(T, deterministicmodel.climate) for T in Tspace]

    policyfig = contourf(mspace, Tspace, E; title = L"Terminal $\bar{\alpha}_{\tau}$", xlabel = L"m", ylabel = L"T", c=:Greens, xlims = extrema(mspace), ylims = extrema(Tspace), clims = (0, 1.2), linewidth = 0.)

    valuefig = contourf(mspace, Tspace, terminalvaluefunction.H; title = L"Terminal value $\bar{H}$", xlabel = L"m", ylabel = L"T", xlims = extrema(mspace), ylims = extrema(Tspace), levels = 21)

    for fig in (policyfig, valuefig)
        plot!(fig, nullcline, Tspace; label = false, c = :black, linewidth = 2.5)
        scatter!(fig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :black)
    end

    plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
end

# Simulate backwards
G = shrink(Gterminal, 0.05, model)
valuefunction = interpolateovergrid(terminalvaluefunction, Gterminal, G)
backwardsimulation!(valuefunction, Δt, model, Gterminal, calibration; t₀ = 295., verbose = 2, withsave = false)

if isinteractive()
    abatement = ε(valuefunction, model, calibration, G)

    policyfig = contourf(mspace, Tspace, abatement; xlabel = L"m", ylabel = L"T", c=:Greens, cmin = 0., xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)
    valuefig = contourf(mspace, Tspace, valuefunction.H; title = L"Initial value $H_0$", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)

    for fig in (policyfig, valuefig)
        plot!(fig, nullcline, Tspace; label = false, c = :white)
        scatter!(fig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :white)
    end

    plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
end