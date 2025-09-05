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

includet("../../src/extend/model.jl")
includet("../../src/valuefunction.jl")
includet("../../src/extend/valuefunction.jl")
includet("../utils/saving.jl")
includet("../markov/utils.jl")
includet("../markov/chain.jl")
includet("../markov/finitedifference.jl")

begin # Construct the model
    calibrationfilepath = "data/calibration.jld2"
    @assert isfile(calibrationfilepath)

    calibrationfile = jldopen(calibrationfilepath, "r+")
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = calibrationfile
    close(calibrationfile)

    damages = Kalkuhl()
    preferences = Preferences()
    economy = Economy()

    threshold = 2.0

    model = if 0 < threshold < Inf
        feedback = Model.updateTᶜ(threshold + hogg.Tᵖ, feedback)
        TippingModel(hogg, preferences, damages, economy, feedback)
    else
        LinearModel(hogg, preferences, damages, economy)
    end

    N = (20, 50)
    Tdomain = hogg.Tᵖ .+ (0., 7.5)
    mdomain = mstable(Tdomain[1] + 0.5, model), mstable(Tdomain[2] - 0.5, model)

    G = RegularGrid(N, (Tdomain, mdomain))
    Δt = 1 / 24
    τ = 0.

    Tspace, mspace = G.ranges
    nullcline = mstable.(Tspace, model)
end;

# Check terminal condition
terminalvaluefunction = ValueFunction(τ, hogg, G, calibration)
richardsonsteadystate!(terminalvaluefunction, Δt, model, G, calibration; verbose = 1, iterations = 10_000, tolerance = Error(1e-5, 1e-6))

if isinteractive()
    αitp = linear_interpolation(G.ranges, terminalvaluefunction.α)
    abatement = [ε(terminalvaluefunction.t.t, x, αitp(x.T, x.m), model, calibration) for x in G]

    policyfig = heatmap(mspace, Tspace, abatement; title = L"Terminal $\bar{\alpha}_{\tau}$", xlabel = L"m", ylabel = L"T", c=:Greens, cmin = 0., cmax = 1., xlims = extrema(mspace), ylims = extrema(Tspace), clims = (0, 1.2))

    valuefig = contourf(mspace, Tspace, terminalvaluefunction.H; title = L"Terminal value $\bar{H}$", xlabel = L"m", ylabel = L"T", xlims = extrema(mspace), ylims = extrema(Tspace), levels = 51)

    for fig in (policyfig, valuefig)
        plot!(fig, nullcline, Tspace; label = false, c = :black, linewidth = 2.5)
        scatter!(fig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :black)
    end

    plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
end

# Simulate backwards
valuefunction = deepcopy(terminalvaluefunction);
if isinteractive() # Backward simulation gif
    Δt⁻¹ = 1 / Δt

    A₀ = constructA(valuefunction, Δt⁻¹, model, G, calibration)
    b₀ = Vector{Float64}(undef, length(G))
    problem = LinearSolve.init(LinearProblem(A₀, b₀), KLUFactorization())

    @gif for iter in 1:120
        updateproblem!(problem, valuefunction, Δt⁻¹, model, G, calibration)
        solve!(problem)
            
        valuefunction.H .= reshape(problem.u, size(G))

        abatement = [ε(valuefunction.t.t, G[i], valuefunction.α[i], model, calibration) for i in CartesianIndices(G)]

        policyfig = contourf(mspace, Tspace, abatement; title = L"Policy $\bar{\alpha}_{%$(valuefunction.t.t)}$", xlabel = L"m", ylabel = L"T", c=:viridis, cmin = 0., cmax = 1., xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)
        valuefig = contourf(mspace, Tspace, valuefunction.H; title = L"Value $H_%$(valuefunction.t.t)$", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)

        for fig in (policyfig, valuefig)
            plot!(fig, nullcline, Tspace; label = false, c = :white)
            scatter!(fig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :white)
        end

        jointfig = plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))

        valuefunction.t.t -= Δt

        jointfig
    end fps = 15
end

backwardsimulation!(valuefunction, Δt, model, G, calibration; t₀ = 295., verbose = 2, withsave = false)

if isinteractive()
    abatement = ε(valuefunction, model, calibration)

    policyfig = contourf(mspace, Tspace, abatement; title = L"Initial $\bar{\alpha}_{0}$", xlabel = L"m", ylabel = L"T", c=:Greens, cmin = 0., xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)
    valuefig = contourf(mspace, Tspace, valuefunction.H; title = L"Initial value $H_0$", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)

    for fig in (policyfig, valuefig)
        plot!(fig, nullcline, Tspace; label = false, c = :white)
        scatter!(fig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :white)
    end

    plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
end