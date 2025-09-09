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

    threshold = Inf
    climate = if 0 < threshold < Inf
        feedback = Model.updateTᶜ(threshold + hogg.Tᵖ, feedback)
        TippingClimate(hogg, decay, feedback)
    else
        LinearClimate(hogg, decay)
    end

    preferences = LogSeparable()

    model = IAM(climate, economy, preferences)
end

begin
    N = (200, 250)
    Tdomain = hogg.Tᵖ .+ (0., 7.)
    mmin = mstable(Tdomain[1] + 0.5, model.climate)
    mmax = mstable(Tdomain[2] - 0.5, model.climate)
    mdomain = (mmin, mmax)
    domains = (Tdomain, mdomain)
    
    G = constructelasticgrid(N, domains, model)
    Δt = 1 / 200
    τ = 300.
end;

# Gif optimisation
if false && isinteractive()
    Δt⁻¹ = 1 / Δt
    valuefunction = ValueFunction(τ, hogg, G, calibration)

    A₀ = constructA!(valuefunction, Δt⁻¹, model, G, calibration)
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

valuefunction = ValueFunction(τ, climate, G, calibration)
steadystate!(valuefunction, Δt, model, G, calibration; verbose = 2, tolerance = Error(1e-3, 1e-3), withnegative = false, timeiterations = 1_000, picarditerations = 2)

if isinteractive()
    Tspace, mspace = G.ranges
    nullcline = [mstable(T, model.climate) for T in Tspace]
    abatement = ε(valuefunction, model, calibration)

    policyfig = contourf(mspace, Tspace, abatement; title = "Drift of CO2e", xlabel = L"m", ylabel = L"T", c=:Greens, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0., clims = (0, 1))
    valuefig = contourf(mspace, Tspace, valuefunction.H; title = "Value Function H", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)

    for fig in (policyfig, valuefig)
        plot!(fig, nullcline, Tspace; label = false, c = :white)
        scatter!(fig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :white)
    end

    jointfig = plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
end