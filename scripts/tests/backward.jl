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
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = climatefile
    close(climatefile)

    decay = ConstantDecay(0.)

    threshold = Inf
    climate = if 0 < threshold < Inf
        feedback = Model.updateTᶜ(threshold, feedback)
        TippingClimate(hogg, decay, feedback)
    else
        LinearClimate(hogg, decay)
    end

    preferences = LogSeparable()
    model = IAM(climate, economy, preferences)
end

begin
    N₁ = 31; N₂ = 31;  # Smaller grid for testing
    N = (N₁, N₂)
    Tdomain = (0.5, 12.)  # Smaller, safer domain
    mmin = mstable(Tdomain[1] + 0.5, model.climate)
    mmax = mstable(Tdomain[2] - 0.5, model.climate)
    mdomain = (mmin, mmax)
    domains = (Tdomain, mdomain)
    withnegative = true

    Gterminal = RegularGrid(N, domains)
    Δt⁻¹ = 10
    Δt = 1 / Δt⁻¹
    τ = 500.
end;

terminalvaluefunction = ValueFunction(τ, climate, Gterminal, calibration)
terminalvaluefunction, result = steadystate!(terminalvaluefunction, Δt, model, Gterminal, calibration; verbose = 1, tolerance = Error(1e-4, 1e-4), timeiterations = 100_000, printstep = 100, withnegative, θ = 1.)

begin # HJB error
    Ā = constructA!(terminalvaluefunction, Δt⁻¹, model, Gterminal, calibration, withnegative)
    b̄ = constructsource(terminalvaluefunction, Δt⁻¹, model, Gterminal, calibration)

    hjb = reshape(Ā * vec(terminalvaluefunction.H) - b̄, size(Gterminal))
end;

if isinteractive()
    Tspace, mspace = Gterminal.ranges
    hjberrorfig = heatmap(mspace, Tspace, hjb; title = "HJB error", xlabel = L"m", ylabel = L"T", c=:Greens, xlims = extrema(mspace), ylims = extrema(Tspace))
end

if isinteractive()
    abatementcolor = cgrad(:RdBu, [0., 2/3, 1])
    E = ε(terminalvaluefunction, model, calibration, Gterminal)
    nullcline = [mstable(T, model.climate) for T in Tspace]

    policyfig = heatmap(mspace, Tspace, E; title = L"Terminal $\bar{\alpha}_{\tau}$", xlabel = L"m", ylabel = L"T", xlims = extrema(mspace), ylims = extrema(Tspace), clims = (0, max(1., maximum(E))), linewidth = 0., color = abatementcolor)

    valuefig = contourf(mspace, Tspace, terminalvaluefunction.H; title = L"Terminal value $\bar{H}$", xlabel = L"m", ylabel = L"T", xlims = extrema(mspace), ylims = extrema(Tspace), levels = 21)

    for fig in (policyfig, valuefig)
        plot!(fig, nullcline, Tspace; label = false, c = :black, linewidth = 2.5)
        scatter!(fig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :black)
    end

    plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
end

# Simulate backwards
G = shrink(Gterminal, (0., 0.05))
valuefunction = interpolateovergrid(terminalvaluefunction, Gterminal, G)

if isinteractive() let # Optimisation Gif
    stencil = makestencil(G)
    A₀ = constructA!(stencil, valuefunction, Δt⁻¹, model, G, calibration, withnegative)
    b₀ = constructsource(valuefunction, Δt⁻¹, model, G, calibration)

    # Initialise the problem
    problem = LinearSolve.init(LinearProblem(A₀, b₀), KLUFactorization())

    Tspace, mspace = G.ranges
    nullcline = [mstable(T, model.climate) for T in Tspace]
    m₀ = log(hogg.M₀ / hogg.Mᵖ)

    abatementcolor = cgrad(:RdBu, [0., 2/3, 1])
    clims = (0, withnegative ? 2.0 : 1.0)

    t₀ = 0.
    nframes = floor(Int, (valuefunction.t.t - t₀) / Δt)
    framestep = max(nframes ÷ 240, 1)

    @gif for iter in 1:nframes
        if (iter % framestep) == 0 print("Iteration $iter / $nframes\r") end

        backwardstep!(problem, stencil, valuefunction, Δt⁻¹, model, G, calibration; withnegative)
        updateovergrid!(valuefunction.H, problem.u, 1.)
        valuefunction.t.t -= Δt

        if any(isnan.(valuefunction.H)) @warn "NaN value in H" end

        valuefig = heatmap(mspace, Tspace, valuefunction.H; title = "Value Function H t = $(valuefunction.t.t)", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)
        
        E = ε(valuefunction, model, calibration, G)
        abatementfig = contourf(mspace, Tspace, E; title = "Abatement t = $(valuefunction.t.t)", xlabel = L"m", ylabel = L"T", color = abatementcolor, xlims = extrema(mspace), ylims = extrema(Tspace), clims = clims, linewidth = 0.)

        for fig in (abatementfig, valuefig)
            plot!(fig, nullcline, Tspace; label = false, c = :white)
            scatter!(fig, [m₀], [hogg.T₀]; label = false, c = :white)
        end

        plot(abatementfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
    end every framestep
end end

backwardsimulation!(valuefunction, Δt, model, G, calibration; t₀ = 150, withnegative, withsave = false, verbose = 1)

if isinteractive()
    e = ε(valuefunction, model, calibration, G)

    policyfig = heatmap(mspace, Tspace, e; xlabel = L"m", ylabel = L"T", xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0., c = cgrad(:RdBu, [0., 2/3, 1]), clims = (0, withnegative ? 2.0 : 1.0))
    valuefig = contourf(mspace, Tspace, valuefunction.H; title = L"Initial value $H_0$", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)

    for fig in (policyfig, valuefig)
        plot!(fig, nullcline, Tspace; label = false, c = :white)
        scatter!(fig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :white)
    end

    plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
end