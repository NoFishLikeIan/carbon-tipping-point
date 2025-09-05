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

    threshold = 4.

    model = if 0 < threshold < Inf
        feedback = Model.updateTᶜ(threshold + hogg.Tᵖ, feedback)
        TippingModel(hogg, preferences, damages, economy, feedback)
    else
        LinearModel(hogg, preferences, damages, economy)
    end

    N = (200, 250)
    Tdomain = hogg.Tᵖ .+ (0., 7.)
    mdomain = mstable(Tdomain[1] + 0.5, model), mstable(Tdomain[2] - 0.5, model)

    G = RegularGrid(N, (Tdomain, mdomain))
    Δt = 1 / 200
    τ = 300.

    if isinteractive()
        Tspace = range(Tdomain[1], Tdomain[2]; length = size(G, 1))
        mspace = range(mdomain[1], mdomain[2]; length = size(G, 2))
        nullcline = mstable.(Tspace, model)
    end
end;

# Gif optimisation
if false && isinteractive()
    Δt⁻¹ = 1 / Δt
    valuefunction = ValueFunction(τ, hogg, G, calibration)

    A₀ = constructA(valuefunction, Δt⁻¹, model, G, calibration)
    b₀ = Vector{Float64}(undef, length(G))
    problem = LinearSolve.init(LinearProblem(A₀, b₀), KLUFactorization())
    γ̄ = γ(valuefunction.t.t, calibration)

    @gif for iter in 1:600
        updateproblem!(problem, valuefunction, Δt⁻¹, model, G, calibration)
        solve!(problem)
            
        valuefunction.H .= reshape(problem.u, size(G))
        Tspace = range(G.domains[1]...; length=size(G, 1))
        mspace = range(G.domains[2]...; length=size(G, 2))

        policyfig = contourf(mspace, Tspace, γ̄ .- valuefunction.α; title = "Drift of CO2e - Iteration $iter", xlabel = L"m", ylabel = L"T", c=:viridis, linewidth = 0, cmin = 0.)
        valuefig = contourf(mspace, Tspace, valuefunction.H; title = "Value Function H - Iteration $iter", xlabel = L"m", ylabel = L"T", c=:viridis, linewidth = 0)

        plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))

    end fps = 30
end

valuefunction = ValueFunction(τ, hogg, G, calibration)
steadystate!(valuefunction, Δt, model, G, calibration; verbose = 2, tolerance = Error(1e-3, 1e-4), withnegative = false, iterations = 1_000)

if isinteractive()
    dm = γ(valuefunction.t.t, calibration) .- valuefunction.α
    dm̄ = maximum(abs, dm)

    policyfig = contourf(mspace, Tspace, dm; title = "Drift of CO2e", xlabel = L"m", ylabel = L"T", c=:coolwarm, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0., clims = (-dm̄, dm̄))
    valuefig = contourf(mspace, Tspace, valuefunction.H; title = "Value Function H", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)

    for fig in (policyfig, valuefig)
        plot!(fig, nullcline, Tspace; label = false, c = :white)
        scatter!(fig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :white)
    end

    jointfig = plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
end