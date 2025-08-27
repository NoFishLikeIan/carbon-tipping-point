using Test, BenchmarkTools, Revise
using Plots, LaTeXStrings

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
includet("../utils/saving.jl")
includet("../utils/logging.jl")
includet("../markov/utils.jl")
includet("../markov/chain.jl")
includet("../markov/terminal.jl")

begin # Construct the model
    calibrationfilepath = "data/calibration.jld2"
    @assert isfile(calibrationfilepath)

    calibrationfile = jldopen(calibrationfilepath, "r+")
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = calibrationfile
    close(calibrationfile)

    damages = Kalkuhl()
    preferences = Preferences()
    economy = Economy()

    threshold = 1.8
    
    model = if 0 < threshold < Inf
        feedback = Model.updateTᶜ(threshold + hogg.Tᵖ, feedback)
        TippingModel(hogg, preferences, damages, economy, feedback)
    else
        LinearModel(hogg, preferences, damages, economy)
    end

    N = (105, 100)
    Tdomain = hogg.Tᵖ .+ (0., 7.)
    mdomain = mstable(Tdomain[1] + 0.5, model), mstable(Tdomain[2] - 0.5, model)

    G = RegularGrid(N, (Tdomain, mdomain))
    Δt = 1 / 12

    if isinteractive()
        Tspace = range(G.domains[1]...; length=size(G, 1))
        mspace = range(G.domains[2]...; length=size(G, 2))
        nullcline = mstable.(Tspace, model)
    end
end;

# Plot optimisation
if isinteractive()
    Δt⁻¹ = 1 / Δt
    valuefunction = ValueFunction(hogg, G, calibration)

    A₀ = constructA(valuefunction, Δt⁻¹, model, G, calibration)
    b₀ = Vector{Float64}(undef, length(G))
    problem = LinearSolve.init(LinearProblem(A₀, b₀), KLUFactorization())

    @gif for iter in 1:600
        updateproblem!(problem, valuefunction, Δt⁻¹, model, G, calibration)
        solve!(problem)
            
        valuefunction.H .= reshape(problem.u, size(G))
        Tspace = range(G.domains[1]...; length=size(G, 1))
        mspace = range(G.domains[2]...; length=size(G, 2))

        policyfig = contourf(mspace, Tspace, valuefunction.α; title = "Abatement Policy - Iteration $iter", xlabel = L"m", ylabel = L"T", c=:viridis, cmin = 0.)
        valuefig = contourf(mspace, Tspace, valuefunction.H; title = "Value Function H - Iteration $iter", xlabel = L"m", ylabel = L"T", c=:viridis)

        plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))

    end fps = 30
end

valuefunction = ValueFunction(hogg, G, calibration)
steadystate!(valuefunction, Δt, model, G, calibration; verbose = 2, tolerance = Error(1e-2, 1e-3))

if isinteractive()
    policyfig = contourf(mspace, Tspace, valuefunction.α; title = "Abatement Policy", xlabel = L"m", ylabel = L"T", c=:viridis, cmin = 0., xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)
    valuefig = contourf(mspace, Tspace, valuefunction.H; title = "Value Function H", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)

    for fig in (policyfig, valuefig)
        plot!(fig, nullcline, Tspace; label = false, c = :white)
        scatter!(fig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :white)
    end

    plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
end