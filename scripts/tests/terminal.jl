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
    damages = WeitzmanGrowth()
    economy = Economy(investments = investments, damages = damages, abatement = abatement)

    # Load climate claibration
    climatepath = joinpath(calibrationpath, "climate.jld2")
    @assert isfile(climatepath) "Climate calibration file not found at $climatepath"
    climatefile = jldopen(climatepath, "r+")
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher, decay = climatefile
    close(climatefile)

    decay = ConstantDecay(0.)

    threshold = Inf #2.5
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
    Tdomain = (0.0, 7.0)  # Smaller, safer domain
    mmin = mstable(Tdomain[1] + 0.5, model.climate)
    mmax = mstable(Tdomain[2] - 0.5, model.climate)
    mdomain = (mmin, mmax)
    domains = (Tdomain, mdomain)
    withnegative = true

    G = constructelasticgrid(N, domains, model)
    Δt⁻¹ = 200.
    Δt = 1 / Δt⁻¹
    τ = 0.
end;

begin # Optimisation Gif
    valuefunction = ValueFunction(τ, climate, G, calibration)
    source = constructsource(valuefunction, Δt⁻¹, model, G, calibration)
    adv = constructadv(valuefunction, model, G, calibration)
    stencil = makestencil(G)
    A₀ = constructA!(stencil, valuefunction, Δt⁻¹, model, G, calibration, withnegative)
    b₀ = source - adv

    # Initialise the problem
    problem = LinearSolve.init(LinearProblem(A₀, b₀), KLUFactorization())

    @gif for iter in 1:1500
        constructadv!(adv, valuefunction, model, G, calibration)
        for _ in 1:3 # Stabilise quadratic approximation
            constructsource!(source, valuefunction, Δt⁻¹, model, G, calibration)
            problem.b .= source - adv  # Use fixed quadratic terms
            
            # Update only the linear parts of the operator (drift + diffusion)
            problem.A = constructA!(stencil, valuefunction, Δt⁻¹, model, G, calibration, withnegative)
            
            sol = solve!(problem)
            if !SciMLBase.successful_retcode(sol)
                throw("Picard step solver failed at time $(valuefunction.t.t)!")
            end
            
            updateovergrid!(valuefunction.H, problem.u, 0.1)  # Much more conservative mixing
        end
        backwardstep!(problem, stencil, valuefunction, Δt⁻¹, model, G, calibration; withnegative)
        updateovergrid!(valuefunction.H, problem.u, 0.3)  # Conservative final update
        
        if any(isnan.(valuefunction.H))
            continue
        end
        Tspace, mspace = G.ranges
        contourf(mspace, Tspace, valuefunction.H; title = "Value Function H iter = $(iter)", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)

        if (iter % 100) == 0
            print("Iteration $iter / 1500.\r")
        end

    end every 10
end

valuefunction, result = steadystate!(valuefunction, Δt, model, G, calibration; verbose = 1, tolerance = Error(1e-3, 1e-4), timeiterations = 1_000, picarditerations = 3, printstep = 100, withnegative)

if isinteractive()
    Tspace, mspace = G.ranges
    nullcline = [mstable(T, model.climate) for T in Tspace]

    e = ε(valuefunction, model, calibration, G)
    
    policyfig = contourf(mspace, Tspace, e; title = "Drift of CO2e", xlabel = L"m", ylabel = L"T", c=:Greens, xlims = extrema(mspace), ylims = extrema(Tspace), clims = (0, max(1, maximum(e))), linewidth = 0.)
    valuefig = contourf(mspace, Tspace, valuefunction.H; title = "Value Function H", xlabel = L"m", ylabel = L"T", c=:viridis, xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0.)

    for fig in (policyfig, valuefig)
        plot!(fig, nullcline, Tspace; label = false, c = :white)
        scatter!(fig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :white)
    end

    jointfig = plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
end