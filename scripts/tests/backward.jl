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
includet("../plotting/utils.jl")
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

    abatement = Model.HambelAbatement

    investments = Investment()
    damages = WeitzmanGrowth() # NoDamageGrowth{Float64}()
    economy = Economy(investments = investments, damages = damages, abatement = abatement)

    # Load climate claibration
    climatepath = joinpath(calibrationpath, "climate.jld2")
    @assert isfile(climatepath) "Climate calibration file not found at $climatepath"
    climatefile = jldopen(climatepath, "r+")
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = climatefile
    close(climatefile)

    decay = ConstantDecay(0.0)
    calibration = ConstantCalibration(0.02)

    threshold = Inf
    climate = if 0 < threshold < Inf
        feedback = Model.updateTᶜ(threshold, feedback)
        TippingClimate(hogg, decay, feedback)
    else
        LinearClimate(hogg, decay)
    end

    preferences = LogSeparable()
    model = IAM(climate, economy, preferences)
    model = determinsticIAM(model) 
end

begin
    N₁ = 51; N₂ = 51;  # Smaller grid for testing
    N = (N₁, N₂)
    Tdomain = (-10., 20.)  # Smaller, safer domain
    mmin = mstable(Tdomain[1], model.climate)
    mmax = mstable(Tdomain[2], model.climate)
    mdomain = (mmin, mmax)
    domains = (Tdomain, mdomain)
    withnegative = true

    Gterminal = RegularGrid(N, domains)
    Δt⁻¹ = 10
    Δt = 1 / Δt⁻¹
    τ = 0.
end;

valuefunction = ValueFunction(τ, climate, Gterminal, calibration)
valuefunction, result = steadystate!(valuefunction, Δt, model, Gterminal, calibration; verbose = 2, tolerance = Error(1e-6, 1e-6), timeiterations = 50, printstep = 100, withnegative = withnegative, θ = 1.)

begin # HJB error
    Ā = constructA!(terminalvaluefunction, Δt⁻¹, model, Gterminal, calibration, withnegative)
    b̄ = constructsource(terminalvaluefunction, Δt⁻¹, model, Gterminal, calibration)

    hjb = reshape(Ā * vec(terminalvaluefunction.H) - b̄, size(Gterminal))

    hjberror = maximum(abs, hjb)
    println("Maximum HJB error: ", hjberror)
end;

if isinteractive()
    Tspace, mspace = Gterminal.ranges
    hjberrorfig = heatmap(mspace, Tspace, abs.(hjb); title = "HJB error", xlabel = L"m", ylabel = L"T", c=:Reds, xlims = extrema(mspace), ylims = extrema(Tspace), clims = (0, hjberror))
end

if isinteractive()
    E = ε(terminalvaluefunction, model, calibration, Gterminal)
    Ē = max(maximum(E), 1)
    nullcline = [mstable(T, model.climate) for T in Tspace]

    policyfig = heatmap(mspace, Tspace, E; title = L"Terminal $\bar{\alpha}_{\tau} / \gamma_t$", xlabel = L"m", ylabel = L"T", xlims = extrema(mspace), ylims = extrema(Tspace), clims = (0, Ē), linewidth = 0., color = abatementcolorbar(Ē))

    valuefig = contourf(mspace, Tspace, terminalvaluefunction.H; title = L"Terminal value $\bar{H}$", xlabel = L"m", ylabel = L"T", xlims = extrema(mspace), ylims = extrema(Tspace), levels = 21)

    for fig in (policyfig, valuefig)
        plot!(fig, nullcline, Tspace; label = false, c = :black, linewidth = 2.5)
        scatter!(fig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :black)
    end

    plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
end

# Simulate backwards
G = shrink(Gterminal, (0., 0.))
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

    t₀ = 25.
    nframes = floor(Int, (valuefunction.t.t - t₀) / Δt)
    framestep = max(nframes ÷ 240, 1)

    @gif for iter in 1:nframes
        if (iter % framestep) == 0 print("Iteration $iter / $nframes\r") end

        backwardstep!(problem, stencil, valuefunction, Δt⁻¹, model, G, calibration; withnegative)
        updateovergrid!(valuefunction.H, problem.u, 1.)
        valuefunction.t.t -= Δt

        if any(isnan.(valuefunction.H)) @warn "NaN value in H" end

        t = round(valuefunction.t.t; digits = 2)
        valuefig = heatmap(mspace, Tspace, valuefunction.H; title = "Value Function H t = $t", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)
        
        E = ε(valuefunction, model, calibration, G)
        Ē = 1.5 # max(maximum(E), 1)
        
        abatementfig = contourf(mspace, Tspace, E; title = "Abatement t = $t", xlabel = L"m", ylabel = L"T", color = abatementcolorbar(Ē), xlims = extrema(mspace), ylims = extrema(Tspace), clims = (0, Ē), linewidth = 0.)

        for fig in (abatementfig, valuefig)
            plot!(fig, nullcline, Tspace; label = false, c = :white)
            scatter!(fig, [m₀], [hogg.T₀]; label = false, c = :white)
        end

        plot(abatementfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
    end every framestep
end end

valuefunction = interpolateovergrid(terminalvaluefunction, Gterminal, G) # Re-initialise the valuefunction
backwardsimulation!(valuefunction, Δt, model, G, calibration; t₀ = 0., withnegative, withsave = false, verbose = 1)

if isinteractive()
    E = ε(valuefunction, model, calibration, G)
    Ē = max(1., maximum(E))

    policyfig = heatmap(mspace, Tspace, E; xlabel = L"m", ylabel = L"T", xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0., c = abatementcolorbar(Ē), clims = (0, Ē))
    valuefig = contourf(mspace, Tspace, valuefunction.H; title = L"Initial value $H_0$", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)

    for fig in (policyfig, valuefig)
        plot!(fig, nullcline, Tspace; label = false, c = :white)
        scatter!(fig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :white)
    end

    plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
end