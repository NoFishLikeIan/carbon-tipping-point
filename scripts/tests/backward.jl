using Test, BenchmarkTools, Revise, UnPack
using Plots, LaTeXStrings
default(c=:viridis, label=false, dpi=180)

using Model, Grid
using Base.Threads
using SciMLBase
using Statistics
using StaticArrays, SparseArrays

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

    threshold = 2.5

    model = if 0 < threshold < Inf
        feedback = Model.updateTᶜ(threshold + hogg.Tᵖ, feedback)
        TippingModel(hogg, preferences, damages, economy, feedback)
    else
        LinearModel(hogg, preferences, damages, economy)
    end

    N = (150, 139)
    Tdomain = hogg.Tᵖ .+ (0., 8.)
    mdomain = mstable(Tdomain[1] + 0.5, model), mstable(Tdomain[2] - 0.5, model)

    G = RegularGrid(N, (Tdomain, mdomain))
    Δt = 1 / 24

    if isinteractive()
        Tspace = range(G.domains[1]...; length=size(G, 1))
        mspace = range(G.domains[2]...; length=size(G, 2))
        nullcline = mstable.(Tspace, model)
    end
end;

valuefunction = ValueFunction(hogg, G, calibration)
steadystate!(valuefunction, Δt, model, G, calibration; verbose = 2, tolerance = Error(1e-2, 1e-3))

if isinteractive()
    policyfig = contourf(mspace, Tspace, valuefunction.α; title = L"Terminal $\bar{\alpha}_{\tau}$", xlabel = L"m", ylabel = L"T", c=:viridis, cmin = 0., xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)
    valuefig = contourf(mspace, Tspace, valuefunction.H; title = L"Terminal value $\bar{H}$", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)

    for fig in (policyfig, valuefig)
        plot!(fig, nullcline, Tspace; label = false, c = :white)
        scatter!(fig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :white)
    end

    plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
end

if isinteractive() # Backward simulation gif
    Δt⁻¹ = 1 / Δt

    A₀ = constructA(valuefunction, Δt⁻¹, model, G, calibration)
    b₀ = Vector{Float64}(undef, length(G))
    problem = LinearSolve.init(LinearProblem(A₀, b₀), KLUFactorization())

    @gif for iter in 1:120
        updateproblem!(problem, valuefunction, Δt⁻¹, model, G, calibration)
        solve!(problem)
            
        valuefunction.H .= reshape(problem.u, size(G))

        policyfig = contourf(mspace, Tspace, valuefunction.α; title = L"Policy $\bar{\alpha}_{%$(valuefunction.t.t)}$", xlabel = L"m", ylabel = L"T", c=:viridis, cmin = 0., xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)
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

backwardsimulation!(valuefunction, Δt, model, G, calibration; t₀ = 0., verbose = 2)

if isinteractive()
    policyfig = contourf(mspace, Tspace, valuefunction.α; title = L"Initial $\bar{\alpha}_{0}$", xlabel = L"m", ylabel = L"T", c=:viridis, cmin = 0., xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)
    valuefig = contourf(mspace, Tspace, valuefunction.H; title = L"Initial value $H_0$", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)

    for fig in (policyfig, valuefig)
        plot!(fig, nullcline, Tspace; label = false, c = :white)
        scatter!(fig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :white)
    end

    plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
end