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

    threshold = 2.5 # Inf
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
    N₁ = 51; N₂ = 51;  # Smaller grid for testing
    N = (N₁, N₂)
    Tdomain = (0.5, 12.)  # Smaller, safer domain
    mmin = mstable(Tdomain[1] + 0.5, model.climate)
    mmax = mstable(Tdomain[2] - 0.5, model.climate)
    mdomain = (mmin, mmax)
    domains = (Tdomain, mdomain)
    withnegative = false

    G = RegularGrid(N, domains)
    Δt⁻¹ = 50
    Δt = 1 / Δt⁻¹
    τ = 500.
end;

valuefunction = ValueFunction(τ, climate, G, calibration)

if false # Optimisation Gif
    source = constructsource(valuefunction, Δt⁻¹, model, G, calibration)
    adv = constructadv(valuefunction, model, G)
    stencil = makestencil(G)
    A₀ = constructA!(stencil, valuefunction, Δt⁻¹, model, G, calibration, withnegative)
    b₀ = source - adv

    # Initialise the problem
    problemdata = (stencil, source, adv)
    problem = LinearSolve.init(LinearProblem(A₀, b₀), KLUFactorization())

    Tspace, mspace = G.ranges
    nullcline = [mstable(T, model.climate) for T in Tspace]
    m₀ = log(hogg.M₀ / hogg.Mᵖ)
    nframes = 5000
    @gif for iter in 1:nframes
        if (iter % (nframes ÷ 100)) == 0 print("Iteration $iter / $nframes.\r") end

        backwardstep!(problem, problemdata, valuefunction, Δt⁻¹, model, G, calibration; withnegative)
        updateovergrid!(valuefunction.H, problem.u, 1.)

        if any(isnan.(valuefunction.H)) @warn "NaN value in H" end

        valuefig = contourf(mspace, Tspace, valuefunction.H; title = "Value Function H ($iter)", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)
        
        emissionsreduction = ε(valuefunction, model, calibration, G)
        abatementfig = contourf(mspace, Tspace, emissionsreduction; title = "Abatement ($iter)", xlabel = L"m", ylabel = L"T", c=:Greens, xlims = extrema(mspace), ylims = extrema(Tspace), clims = (0, max(1, maximum(emissionsreduction))), linewidth = 0.)

        for fig in (abatementfig, valuefig)
            plot!(fig, nullcline, Tspace; label = false, c = :white)
            scatter!(fig, [m₀], [hogg.T₀]; label = false, c = :white)
        end

        plot(abatementfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))

    end every 100
end

valuefunction, result = steadystate!(valuefunction, Δt, model, G, calibration; verbose = 1, tolerance = Error(1e-4, 1e-4), timeiterations = 100_000, printstep = 100, withnegative, θ = 1.)

if isinteractive()
    Gfig = shrink(G, (0., 0.05))
    Vfig = interpolateovergrid(valuefunction, G, Gfig)

    Tspace, mspace = Gfig.ranges
    nullcline = [mstable(T, model.climate) for T in Tspace]
    m₀ = log(hogg.M₀ / hogg.Mᵖ)
    
    e = ε(Vfig, model, calibration, Gfig)
    
    policyfig = contourf(mspace, Tspace, e; title = "Emissions abated τ = $τ", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), clims = (0, max(1, maximum(e))), linewidth = 0, levels = 101)
    valuefig = contourf(mspace, Tspace, Vfig.H; title = "Value Function H τ = $τ", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)

    for fig in (policyfig, valuefig)
        plot!(fig, nullcline, Tspace; label = false, c = :white)
        scatter!(fig, [m₀], [hogg.T₀]; label = false, c = :white)
    end

    jointfig = plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
end
