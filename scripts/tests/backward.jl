using Test, BenchmarkTools, Revise, UnPack
using Plots, LaTeXStrings
default(c=:viridis, label=false, dpi=180)

using Model, Grid
using Base.Threads
using SciMLBase
using Statistics
using StaticArrays, SparseArrays
using Interpolations, DataStructures

using LinearSolve, LinearAlgebra

using JLD2, UnPack
using Dates, Printf

includet("../../src/valuefunction.jl")
includet("../../src/extend/model.jl")
includet("../../src/extend/grid.jl")
includet("../../src/extend/valuefunction.jl")
includet("../utils/saving.jl")
includet("../utils/simulating.jl")
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

    investments = Investment()
    damages = BurkeHsiangMiguel() # WeitzmanGrowth()
    economy = Economy(investments = investments, damages = damages, abatement = abatement)

    # Load climate claibration
    climatepath = joinpath(calibrationpath, "climate.jld2")
    @assert isfile(climatepath) "Climate calibration file not found at $climatepath"
    climatefile = jldopen(climatepath, "r+")
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = climatefile
    close(climatefile)

    decay = ConstantDecay(0.)
    threshold = 2.
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
    N₁ = 100; N₂ = 101;
    N = (N₁, N₂)
    Tmin = 0.; Tmax = 8.;
    mmin = mstable(Tmin + 0.1, model.climate)
    mmax = mstable(Tmax - 0.1, model.climate)
    
    Tdomain = (Tmin, Tmax)
    mdomain = (mmin, mmax)
    
    domains = (Tdomain, mdomain)
    withnegative = true

    G = RegularGrid(N, domains)
    Δt⁻¹ = 24.
    Δt = 1 / Δt⁻¹
    τ = 500.
end;

valuefunction = ValueFunction(τ, climate, G, calibration)

equilibriumsteadystate!(valuefunction, Δt, linearIAM(model), G, calibration; verbose = 1, timeiterations = 100_000, printstep = 10_000, tolerance = Error(1e-8, 1e-8))
eqvaluefunction = deepcopy(valuefunction)

steadystate!(valuefunction, Δt, model, G, calibration; timeiterations = 10_000, printstep = 1_000, verbose = 1, tolerance = Error(1e-7, 1e-8))

let # HJB error
    n = prod(size(G))
    stencilT, stencilm = makestencil(G)
    constructDᵀ!(stencilT, model, G)
    constructDᵐ!(stencilm, valuefunction, model, G, calibration, withnegative)
    b̄ = constructsource(valuefunction, Δt⁻¹, model, G, calibration)
    Sᵨ = (preferences.ρ + Δt⁻¹) * I
    R = Sᵨ - sparse(stencilT[1], stencilT[2], stencilT[3], n, n)
    Ā = R - sparse(stencilm[1], stencilm[2], stencilm[3], n, n)

    hjb = reshape(Ā * vec(valuefunction.H) - b̄, size(G))

    hjberror = maximum(abs, hjb)
    println("Maximum HJB error: ", hjberror)

    if isinteractive()
        Tspace, mspace = G.ranges
        hjberrorfig = heatmap(mspace, Tspace, abs.(hjb); title = "HJB error", xlabel = L"m", ylabel = L"T", c=:Reds, xlims = extrema(mspace), ylims = extrema(Tspace), clims = (0, hjberror))
    end
end

if isinteractive()
    Tspace, mspace = G.ranges
    E = ε(valuefunction, model, calibration, G)
    Ē = max(maximum(E), 1)
    nullcline = [mstable(T, model.climate) for T in Tspace]

    policyfig = heatmap(mspace, Tspace, E; title = L"Terminal $\bar{\alpha}_{\tau} / \gamma_t$", xlabel = L"m", ylabel = L"T", xlims = extrema(mspace), ylims = extrema(Tspace), clims = (0, Ē), linewidth = 0., color = abatementcolorbar(Ē))

    valuefig = contourf(mspace, Tspace, valuefunction.H; title = L"Terminal value $\bar{H}$", xlabel = L"m", ylabel = L"T", xlims = extrema(mspace), ylims = extrema(Tspace), levels = 21)

    for fig in (policyfig, valuefig)
        plot!(fig, nullcline, Tspace; label = false, c = :black, linewidth = 2.5)
        scatter!(fig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :black)
    end

    plot(policyfig, valuefig; layout=(1,2), size = 600 .* (2√2, 1))
end

# Simulate backwards
valuefunctiontraj = backwardsimulation!(valuefunction, Δt, model, G, calibration; t₀ = 0., withnegative, withsave = false, verbose = 1, storetrajectory = true)

if isinteractive()
    Ē = 1.5
    anim = @animate for (t, valuefunction) in valuefunctiontraj
        E = ε(valuefunction, model, calibration, G)

        policyfig = heatmap(mspace, Tspace, E; xlabel = L"m", ylabel = L"T", title = "Time $(round(t, digits = 2))", xlims = extrema(mspace), ylims = extrema(Tspace), linewidth = 0., c = abatementcolorbar(Ē), clims = (0, Ē))

        plot!(policyfig, nullcline, Tspace; label = false, c = :white)
        scatter!(policyfig, [log(hogg.M₀ / hogg.Mᵖ)], [hogg.T₀]; label = false, c = :white)

        policyfig
    end

    gif(anim, fps = 30)
end


# Test simulation
using DifferentialEquations

Hitp, αitp = buildinterpolations(valuefunctiontraj, G);
X₀ = SVector(hogg.T₀, log(hogg.M₀ / hogg.Mᵖ), 0., 0., 0., 0.)
simulationparameters = (model, calibration, αitp);

prob = ODEProblem(F, X₀, (0., 500.), simulationparameters)
sol = solve(prob, Tsit5())

# Compute α and ε along the solution trajectory `sol`
begin
    ts = sol.t
    n = length(ts)
    αsol = Vector{Float64}(undef, n)
    εsol = Vector{Float64}(undef, n)
    γsol = Vector{Float64}(undef, n)
    δsol = Vector{Float64}(undef, n)

    for (i, t) in enumerate(ts)
        u = sol.u[i]
        T, m = @view u[1:2]
        α = αitp(T, m, t)
        αsol[i] = α
        εsol[i] = ε(t, Point(T, m), α, model, calibration)
        γsol[i] = γ(t, calibration)
        # δₘ takes absolute CO₂e concentration M (ppm), convert from log-m
        M = exp(m) * model.climate.hogg.Mᵖ
        δsol[i] = δₘ(M, model.climate.decay)
    end
end

# Plot α and ε through time and save figures
begin
    # Derived series for checks/combined plotting
    dm_dt = γsol .- αsol
    ε_calc = αsol ./ (γsol .+ δsol)

    p1 = plot(ts, αsol; xlabel = "t", ylabel = L"\alpha, \gamma", title = "α and γ (and dm/dt) along solution", label = "α", lw=2)
    plot!(p1, ts, γsol; label = "γ", lw=2, ls = :dash)
    plot!(p1, ts, dm_dt; label = "dm/dt = γ - α", lw=1, ls = :dot)

    p2 = plot(ts, εsol; xlabel = "t", ylabel = L"\epsilon", title = "ε and reconstructed ε = α / (γ + δₘ)", label = "ε (from VF)", lw=2)
    plot!(p2, ts, ε_calc; label = "ε_calc = α/(γ+δₘ)", lw=2, ls=:dash)

    p3 = plot(ts, δsol; xlabel = "t", ylabel = L"\delta_m", title = "δ_m(t) along solution", label = "δ_m", lw=2)
    plot!(p3, ts, γsol .+ δsol; label = "γ + δ_m", lw=2, ls=:dash)

    p4 = plot(ts, dm_dt; xlabel = "t", ylabel = L"dm/dt", title = "dm/dt along solution", label = "dm/dt", lw=2)

    p = plot(p1, p2, p3, p4; layout = (2, 2), size = (1000, 800))
end

