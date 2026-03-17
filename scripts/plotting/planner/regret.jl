using Revise
using JLD2, UnPack, DataStructures
using FastClosures
using Base.Threads
using SciMLBase
using Statistics
using SciMLBase, DifferentialEquations, DiffEqBase, StochasticDiffEq
using Interpolations, Dierckx, FastChebInterp
using StaticArrays, SparseArrays
using LinearAlgebra, LinearSolve
using ForwardDiff

using Model, Grid
using Random; Random.seed!(11148705);

using Plots, PGFPlotsX, Contour
using LaTeXStrings, Printf
using Colors, ColorSchemes

# pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE,
    raw"\usepgfplotslibrary{fillbetween}",
    raw"\usetikzlibrary{patterns}",
    raw"\usepackage{siunitx}",
    raw"\DeclareSIUnit{\ppm}{p.p.m.}",
    raw"\DeclareSIUnit{\CO}{\,CO_2e}",
    raw"\DeclareSIUnit{\output}{trillion US\mathdollar / year}",
    raw"\DeclareSIUnit{\shortoutput}{tr US\mathdollar / y}",
)

includet("../utils.jl")
includet("../../../src/valuefunction.jl")
includet("../../../src/extend/model.jl")
includet("../../../src/extend/grid.jl")
includet("../../../src/extend/valuefunction.jl")
includet("../../../src/regret.jl")

includet("../../utils/saving.jl")
includet("../../utils/simulating.jl")
includet("../../utils/loading.jl")
includet("../../utils/approximate.jl")

includet("../../markov/chain.jl")
includet("../../markov/finitedifference.jl")
includet("../../regret/chain.jl")
includet("../../regret/finitedifference.jl")

## Define plotting constants
DATAPATH = "data"
SIMDATAPATH = "data/simulation"
calibrationpath = joinpath(DATAPATH, "calibration")
SAVEFIG = true;
PLOTPATH = "../job-market-paper/jeem/plots"
plotpath = joinpath(PLOTPATH, "regret")
if !isdir(plotpath) mkpath(plotpath) end

regretpolicypath = joinpath(DATAPATH, "regret", "policy.jld2")
@assert isfile(regretpolicypath) "Missing regret policy file at $regretpolicypath"
JLD2.@load regretpolicypath weights policybasis r Tᶜ
weights = SVector{size(policybasis)}(weights)

damagetype = BurkeHsiangMiguel
withnegative = true
abatementtype = withnegative ? "negative" : "constrained"
@assert isdir(SIMDATAPATH)

horizon = 100.
tspan = (0., horizon)

function stablebranches(T̄, ts)
    lowidx = findlast(Tₜ -> length(Tₜ) > 1, T̄)
    T̄low, tlow = if !isnothing(lowidx)
        first.(T̄[1:lowidx]), ts[1:lowidx]
    else
        first.(T̄), ts
    end

    highidx = findfirst(Tₜ -> length(Tₜ) > 1, T̄)
    T̄high, thigh = if !isnothing(highidx)
        last.(T̄[highidx:end]), ts[highidx:end]
    else
        Float64[], Float64[]
    end

    return T̄low, tlow, T̄high, thigh
end

"Compute ε quantile paths from quantiles of (T, m) states and a policy evaluator."
function epsilonpaths(paths, model, calibration, αfun)
    epaths = NTuple{3, Float64}[]
    for (t, u) in zip(paths.t, paths.u)
        Tq, mq = u[1:2]
        εq = ntuple(i -> ε(t, Point(Tq[i], mq[i]), αfun(Tq[i], mq[i], t), model, calibration), 3)
        push!(epaths, εq)
    end

    return epaths
end

## Read available files
simulationfiles = listfiles(SIMDATAPATH)
@assert !isempty(simulationfiles) "No simulation files found under $SIMDATAPATH"

maximumthreshold = Inf
modelfiles = String[]
for (i, filepath) in enumerate(simulationfiles)
    print("Reading $i / $(length(simulationfiles))\r")
    model, _ = loadproblem(filepath)
    abatementdir = splitpath(filepath)[end - 1]

    isdamage = model.economy.damages isa damagetype
    isabatement = (abatementdir == abatementtype)
    isthreshold = model.climate isa LinearClimate || model.climate.feedback.Tᶜ ≤ maximumthreshold

    if isdamage && isabatement && isthreshold
        push!(modelfiles, filepath)
    end
end

println("$(length(modelfiles)) models detected.")

## Import available files
localmodels = IAM[]
interpolations = Dict{IAM, NTuple{2, Interpolations.Extrapolation}}()
_, G = loadproblem(modelfiles[1]) # Assume all models share G

for (i, filepath) in enumerate(modelfiles)
    print("Loading $i / $(length(modelfiles))\r")
    values, model, G = loadtotal(filepath; tspan=(0, 1.01horizon))
    interpolations[model] = buildinterpolations(values, G)
    push!(localmodels, model)
end

sort!(by = m -> m.climate, localmodels, rev = true)
@assert length(localmodels) >= 2 "Need at least two models to compare Linear and Tipping cases."

const extremamodels = (localmodels[1], localmodels[end])
const extremalabels = ("Linear", "Tipping")

## Load calibration
abatementpath = joinpath(calibrationpath, "abatement.jld2")
@assert isfile(abatementpath) "Abatement calibration file not found at $abatementpath"
abatementfile = jldopen(abatementpath, "r+")
@unpack abatement = abatementfile
close(abatementfile)

investments = Investment()
damages = BurkeHsiangMiguel() # WeitzmanGrowth()
economy = Economy(investments = investments, damages = damages, abatement = abatement)

preferences = LogSeparable()

climatepath = joinpath(calibrationpath, "climate.jld2")
@assert isfile(climatepath) "Climate calibration file not found at $climatepath"
climatefile = jldopen(climatepath, "r+")
@unpack calibration, hogg, feedback, decay = climatefile
close(climatefile)

## Shared plotting/simulation constants
PALETTE = colorschemes[:grays]
colors = reverse(get(PALETTE, range(0, 0.6; length=length(extremamodels))))
LINE_WIDTH = 2.5
QS = (0.1, 0.5, 0.9)
NTRAJECTORIES = 10_000

Tspace, mspace = G.ranges
temperatureticks = makedeviationtickz(Tspace[1], Tspace[end]; step=1, digits=2)
yearticks = 0:20:horizon

T₀ = hogg.T₀
m₀ = log(hogg.M₀ / hogg.Mᵖ)
X₀ = SVector(T₀, m₀, 0.)
u₀ = SVector(X₀..., 0., 0., 0.)

mnpprob = ODEProblem((_, calibration, t) -> γ(t, calibration), m₀, (0, horizon), calibration)
mnp = solve(mnpprob, Tsit5())

## Simulate trajectories under optimal and regret controls
optimalsims = Dict{IAM, EnsembleSolution}()
regretsims = Dict{IAM, EnsembleSolution}()

for (i, model) in enumerate(extremamodels)
    _, αitp = interpolations[model]

    optimalparameters = (model, calibration, αitp)
    regretparameters = (model, calibration, policybasis, weights)

    optimalproblem = SDEProblem(F, noise, u₀, tspan, optimalparameters)
    regretproblem = SDEProblem(F, noise, u₀, tspan, regretparameters)

    optimalsims[model] = solve(EnsembleProblem(optimalproblem); trajectories = NTRAJECTORIES)
    regretsims[model] = solve(EnsembleProblem(regretproblem); trajectories = NTRAJECTORIES)

    println("Simulated model $i / $(length(extremamodels))")
end

## Policy-rule comparison: ε(M_t^{np}) for optimal vs regret
let
    mediantspan = 0:10:60
    years = 2020 .+ Int.(mediantspan)
    mmedianpath = mnp(mediantspan).u
    Mmedianpath = @. hogg.Mᵖ * exp(mmedianpath)
    mmin, mmax = extrema(mmedianpath)

    Mtickslabels = [
        L"\footnotesize $%$M$ \\ \footnotesize ($%$y$)"
        for (M, y) in zip(round.(Int, Mmedianpath), years)
    ]

    ytick = 0.4:0.2:1.4
    yticklabels = [@sprintf("\\footnotesize %.0f\\%%", 100y) for y in ytick]

    bandx = [mmin, mmax]
    bandy = [1.0, 1.45]
    bandcoords = vcat([(x, bandy[1]) for x in bandx], [(x, bandy[2]) for x in reverse(bandx)])
    bandpoly = @pgf Plot({fill = "gray", opacity = 0.2, draw = "none", forget_plot}, Coordinates(bandcoords))

    policyfig = @pgf Axis({
        ymin = 0.35, ymax = 1.45,
        ytick = ytick, yticklabels = yticklabels,
        xlabel = L"\footnotesize \si{\CO} concentration $M_t^{\textrm{np}} \; [\si{\ppm}]$",
        ylabel = L"\footnotesize Fraction of abated emissions $\varepsilon_t$",
        xtick = mmedianpath, xticklabels = Mtickslabels,
        xmin = mmin, xmax = mmax,
        width = raw"0.8\linewidth", xticklabel_style = {align = "center"},
        grid = "both",
        legend_pos = "north west"
    })

    push!(policyfig, bandpoly)

    timepoints = 0:0.1:horizon

    # Optimal policy for each model
    for (k, model) in enumerate(extremamodels)
        _, αitp = interpolations[model]

        T̄ = [Tstable(mnp(t), model.climate) for t in timepoints]
        T̄low, tlow, T̄high, thigh = stablebranches(T̄, timepoints)

        mlow = [mnp(t) for t in tlow]
        mhigh = [mnp(t) for t in thigh]

        εoptlow = [ε(t, Point(T, m), αitp(T, m, t), model, calibration) for (T, m, t) in zip(T̄low, mlow, tlow)]

        optlow = @pgf Plot({line_width = LINE_WIDTH, color = colors[k], solid}, Coordinates(mlow, εoptlow))
        push!(policyfig, optlow)
        push!(policyfig, LegendEntry("\\footnotesize $(extremalabels[k]) optimal"))

        if !isempty(T̄high)
            εopthigh = [ε(t, Point(T, m), αitp(T, m, t), model, calibration) for (T, m, t) in zip(T̄high, mhigh, thigh)]
            opthigh = @pgf Plot({line_width = LINE_WIDTH, color = colors[k], forget_plot}, Coordinates(mhigh, εopthigh))
            tippingmarker = @pgf Plot({
                mark_options = {fill = colors[k]}, only_marks, forget_plot
            }, Coordinates(mhigh[[1]], εopthigh[[1]]))
            push!(policyfig, opthigh, tippingmarker)
        end
    end

    # Robust policy (model-independent α; evaluated along linear model's stable branch)
    let model = extremamodels[1]
        T̄ = [Tstable(mnp(t), model.climate) for t in timepoints]
        T̄low, tlow, T̄high, thigh = stablebranches(T̄, timepoints)

        mlow = [mnp(t) for t in tlow]
        mhigh = [mnp(t) for t in thigh]

        εreglow = [ε(t, Point(T, m), weightedpolicy(Point(T, m), t, weights, policybasis), model, calibration) for (T, m, t) in zip(T̄low, mlow, tlow)]

        reglow = @pgf Plot({line_width = LINE_WIDTH, color = "black", dashdotted}, Coordinates(mlow, εreglow))
        push!(policyfig, reglow)
        push!(policyfig, LegendEntry("\\footnotesize Robust"))

        if !isempty(T̄high)
            εreghigh = [ε(t, Point(T, m), weightedpolicy(Point(T, m), t, weights, policybasis), model, calibration) for (T, m, t) in zip(T̄high, mhigh, thigh)]
            reghigh = @pgf Plot({line_width = LINE_WIDTH, color = "black", densely_dotted, forget_plot}, Coordinates(mhigh, εreghigh))
            tippingmarker = @pgf Plot({
                mark_options = {fill = colors[k]}, only_marks, forget_plot
            }, Coordinates(mhigh[[1]], εopthigh[[1]]))
            push!(policyfig, opthigh, tippingmarker)
        end
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "regret-policyfig.tikz"), policyfig; include_preamble = true)
    end

    policyfig
end

## Dynamics comparison: state trajectories (M_t, T_t) for optimal vs regret
controlledtemperatureticks = makedeviationtickz(1., 3.; step=1, digits=2)

let
    statefig = @pgf GroupPlot({
        group_style = {
            group_size = "$(length(extremamodels)) by 2",
            horizontal_sep = raw"2em"
    }})

    medianopts = @pgf {line_width = LINE_WIDTH}
    confidenceopts = @pgf {draw = "none", forget_plot}
    figopts = @pgf {width = raw"0.49\linewidth", height = raw"0.34\linewidth", grid = "both", xmin = 0, xmax = horizon}

    yearlytime = 0:horizon

    # Carbon concentration row
    for (k, model) in enumerate(extremamodels)
        optimalpaths = EnsembleAnalysis.timeseries_point_quantile(optimalsims[model], QS, yearlytime)
        regretpaths = EnsembleAnalysis.timeseries_point_quantile(regretsims[model], QS, yearlytime)

        Mopt = [@. hogg.Mᵖ * exp(m) for m in getindex.(optimalpaths.u, 2)]
        Mreg = [@. hogg.Mᵖ * exp(m) for m in getindex.(regretpaths.u, 2)]

        fillopt = @pgf {fill = colors[k], opacity = 0.15, forget_plot}
        fillreg = @pgf {fill = colors[k], opacity = 0.08, forget_plot}

        lowopt = "Mlowopt$(k)"
        highopt = "Mhighopt$(k)"
        lowreg = "Mlowreg$(k)"
        highreg = "Mhighreg$(k)"

        Mmedianopt = @pgf Plot({medianopts..., color = colors[k], solid}, Coordinates(yearlytime, getindex.(Mopt, 2)))
        Mloweropt = @pgf Plot({confidenceopts..., color = colors[k], name_path = lowopt}, Coordinates(yearlytime, getindex.(Mopt, 1)))
        Mupperopt = @pgf Plot({confidenceopts..., color = colors[k], name_path = highopt}, Coordinates(yearlytime, getindex.(Mopt, 3)))
        Mfillopt = @pgf Plot(fillopt, "fill between [of=$lowopt and $highopt]")

        Mmedianreg = @pgf Plot({medianopts..., color = colors[k], dashdotted}, Coordinates(yearlytime, getindex.(Mreg, 2)))
        Mlowerreg = @pgf Plot({confidenceopts..., color = colors[k], name_path = lowreg}, Coordinates(yearlytime, getindex.(Mreg, 1)))
        Mupperreg = @pgf Plot({confidenceopts..., color = colors[k], name_path = highreg}, Coordinates(yearlytime, getindex.(Mreg, 3)))
        Mfillreg = @pgf Plot(fillreg, "fill between [of=$lowreg and $highreg]")

        labelopts = @pgf k > 1 ? {
            yticklabel = raw"\empty"
        } : {
            ylabel = L"Concentration $M_t \; [\si{ppm}]$"
        }

        legendopt = k > 1 ? (LegendEntry(raw"\footnotesize Optimal"),) : ()
        legendreg = k > 1 ? (LegendEntry(raw"\footnotesize Robust"),) : ()

        @pgf push!(statefig, {figopts...,
                xticklabel = raw"\empty",
                ymin = hogg.M₀,
                ymax = 600.,
                title = extremalabels[k],
                labelopts...
            }, Mmedianopt, legendopt..., Mloweropt, Mupperopt, Mfillopt, Mmedianreg, legendreg..., Mlowerreg, Mupperreg, Mfillreg)
    end

    # Temperature row
    for (k, model) in enumerate(extremamodels)
        optimalpaths = EnsembleAnalysis.timeseries_point_quantile(optimalsims[model], QS, yearlytime)
        regretpaths = EnsembleAnalysis.timeseries_point_quantile(regretsims[model], QS, yearlytime)

        Topt = first.(optimalpaths.u)
        Treg = first.(regretpaths.u)

        fillopt = @pgf {fill = colors[k], opacity = 0.15, forget_plot}
        fillreg = @pgf {fill = colors[k], opacity = 0.08, forget_plot}

        lowopt = "Tlowopt$(k)"
        highopt = "Thighopt$(k)"
        lowreg = "Tlowreg$(k)"
        highreg = "Thighreg$(k)"

        Tmedianopt = @pgf Plot({medianopts..., color = colors[k], solid}, Coordinates(yearlytime, getindex.(Topt, 2)))
        Tloweropt = @pgf Plot({confidenceopts..., color = colors[k], name_path = lowopt}, Coordinates(yearlytime, getindex.(Topt, 1)))
        Tupperopt = @pgf Plot({confidenceopts..., color = colors[k], name_path = highopt}, Coordinates(yearlytime, getindex.(Topt, 3)))
        Tfillopt = @pgf Plot(fillopt, "fill between [of=$lowopt and $highopt]")
        Tloweroptline = @pgf Plot({line_width = 0.6, color = colors[k], solid, forget_plot}, Coordinates(yearlytime, getindex.(Topt, 1)))
        Tupperoptline = @pgf Plot({line_width = 0.6, color = colors[k], solid, forget_plot}, Coordinates(yearlytime, getindex.(Topt, 3)))

        Tmedianreg = @pgf Plot({medianopts..., color = colors[k], dashdotted}, Coordinates(yearlytime, getindex.(Treg, 2)))
        Tlowerreg = @pgf Plot({confidenceopts..., color = colors[k], name_path = lowreg}, Coordinates(yearlytime, getindex.(Treg, 1)))
        Tupperreg = @pgf Plot({confidenceopts..., color = colors[k], name_path = highreg}, Coordinates(yearlytime, getindex.(Treg, 3)))
        Tfillreg = @pgf Plot(fillreg, "fill between [of=$lowreg and $highreg]")
        Tlowerregline = @pgf Plot({line_width = 0.6, color = colors[k], dashed, forget_plot}, Coordinates(yearlytime, getindex.(Treg, 1)))
        Tupperregline = @pgf Plot({line_width = 0.6, color = colors[k], dashed, forget_plot}, Coordinates(yearlytime, getindex.(Treg, 3)))

        labelopts = @pgf k > 1 ? {
            yticklabel = raw"\empty"
        } : {
            ytick = controlledtemperatureticks[1],
            yticklabels = controlledtemperatureticks[2],
            ylabel = raw"Temperature $T_t$"
        }

        @pgf push!(statefig, {figopts...,
                ymin = minimum(controlledtemperatureticks[1]),
                ymax = maximum(controlledtemperatureticks[1]),
                xtick = yearticks,
                xticklabels = 2020 .+ Int.(yearticks),
                xticklabel_style = {rotate = 45},
                xlabel = "Year",
                labelopts...
            }, Tmedianopt, Tloweropt, Tupperopt, Tfillopt, Tloweroptline, Tupperoptline, Tmedianreg, Tlowerreg, Tupperreg, Tfillreg, Tlowerregline, Tupperregline)
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "regret-simfig-state.tikz"), statefig; include_preamble = true)
    end

    statefig
end

## Dynamics comparison: abatement trajectory (ε_t) for optimal vs regret
let
    abatementfig = @pgf GroupPlot({
        group_style = {
            group_size = "$(length(extremamodels)) by 1",
            horizontal_sep = raw"2em"
    }})

    medianopts = @pgf {line_width = LINE_WIDTH}
    confidenceopts = @pgf {draw = "none", forget_plot}
    figopts = @pgf {width = raw"0.49\linewidth", height = raw"0.34\linewidth", grid = "both", xmin = 0, xmax = horizon}

    yearlytime = 0:horizon
    εtick = 0.4:0.2:1.4
    εticklabels = [@sprintf("\\footnotesize %.0f\\%%", 100y) for y in εtick]

    for (k, model) in enumerate(extremamodels)
        _, αitp = interpolations[model]

        optimalpaths = EnsembleAnalysis.timeseries_point_quantile(optimalsims[model], QS, yearlytime)
        regretpaths = EnsembleAnalysis.timeseries_point_quantile(regretsims[model], QS, yearlytime)

        αoptimal = (T, m, t) -> αitp(T, m, t)
        αregret = (T, m, t) -> weightedpolicy(Point(T, m), t, weights, policybasis)

        εopt = epsilonpaths(optimalpaths, model, calibration, αoptimal)
        εreg = epsilonpaths(regretpaths, model, calibration, αregret)

        optnetzero = yearlytime[findfirst(ε -> ε ≥ 1., getindex.(εopt, 2))] + 2020
        regnetzero = yearlytime[findfirst(ε -> ε ≥ 1., getindex.(εreg, 2))] + 2020

        println("Net zero: optimal $optnetzero, robust $regnetzero")

        fillopt = @pgf {fill = colors[k], opacity = 0.15, forget_plot}
        fillreg = @pgf {fill = colors[k], opacity = 0.08, forget_plot}

        lowopt = "Elowopt$(k)"
        highopt = "Ehighopt$(k)"
        lowreg = "Elowreg$(k)"
        highreg = "Ehighreg$(k)"

        εmedianopt = @pgf Plot({medianopts..., color = colors[k], solid}, Coordinates(yearlytime, getindex.(εopt, 2)))
        εloweropt = @pgf Plot({confidenceopts..., color = colors[k], name_path = lowopt}, Coordinates(yearlytime, getindex.(εopt, 1)))
        εupperopt = @pgf Plot({confidenceopts..., color = colors[k], name_path = highopt}, Coordinates(yearlytime, getindex.(εopt, 3)))
        εfillopt = @pgf Plot(fillopt, "fill between [of=$lowopt and $highopt]")

        εmedianreg = @pgf Plot({medianopts..., color = colors[k], dashdotted}, Coordinates(yearlytime, getindex.(εreg, 2)))
        εlowerreg = @pgf Plot({confidenceopts..., color = colors[k], name_path = lowreg}, Coordinates(yearlytime, getindex.(εreg, 1)))
        εupperreg = @pgf Plot({confidenceopts..., color = colors[k], name_path = highreg}, Coordinates(yearlytime, getindex.(εreg, 3)))
        εfillreg = @pgf Plot(fillreg, "fill between [of=$lowreg and $highreg]")

        bandx = (0, horizon)
        bandy = [1.0, 1.45]
        bandcoords = vcat([(x, bandy[1]) for x in bandx], [(x, bandy[2]) for x in reverse(bandx)])
        bandpoly = @pgf Plot({fill = "gray", opacity = 0.2, draw = "none", forget_plot}, Coordinates(bandcoords))

        figticks = yearticks[1:(k > 1 ? end : end - 1)]

        labelopts = @pgf k > 1 ? {
            yticklabel = raw"\empty"
        } : {
            ylabel = raw"Abated emissions fraction $\varepsilon_t$",
            ytick = εtick,
            yticklabels = εticklabels
        }

        legendopts = @pgf k > 1 ? {
            legend_pos = "south east"
        } : {
        }

        legendopt = k > 1 ? (LegendEntry(raw"\footnotesize Optimal"),) : ()
        legendreg = k > 1 ? (LegendEntry(raw"\footnotesize Regret"),) : ()

        @pgf push!(abatementfig, {figopts...,
                ymin = 0.35,
                ymax = 1.45,
                xtick = figticks,
                xticklabels = 2020 .+ Int.(figticks),
                xticklabel_style = {rotate = 45},
                xlabel = "Year",
                title = extremalabels[k],
            labelopts...,
            legendopts...
            }, εmedianopt, legendopt..., εloweropt, εupperopt, εfillopt, εmedianreg, legendreg..., εlowerreg, εupperreg, εfillreg, bandpoly)
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "regret-simfig-abatement.tikz"), abatementfig; include_preamble = true)
    end

    abatementfig
end

# Uncertainty premium
## Extract full information value function & compute SCCₜ
simpath = "data/simulation"
paths = loadsimulationpaths(simpath; exclude = ["terminal", "linear"])
threshold = round(Tᶜ, digits = 1)
path = paths[threshold]
values, model, G = loadtotal(path)
Hitp, αitp = buildinterpolations(values, G);


## Extract welfare of φʳ
G = coarse(G, (2, 2))
τ = 150.; Δt = 1 / 8
climate = TippingClimate(hogg, ConstantDecay(0.), updatethreshold(threshold, feedback))

valuefunction = ValueFunction(τ, climate, G, calibration)
valuefunctiontraj = backwardsimulation!(valuefunction, weights, Δt, model, G, calibration, policybasis; verbose = 1, storetrajectory = true)

Hʳitp, _ = buildinterpolations(valuefunctiontraj, G)

## Simulate state variables under the two policies
noiseprocess = WienerProcess(0., SVector{3}(zeros(3)))
jointx₀ = SVector{6}(X₀..., X₀...)

counterfactualregretparams = ((model, calibration, αitp), (model, calibration, policybasis, weights));

function Fjoint(jointx::V, p, t) where V
    x⁺ = @view jointx[1:3]
    xʳ = @view jointx[4:6]

    dx⁺ = F(SVector{3}(x⁺), p[1], t)
    dxʳ = F(SVector{3}(xʳ), p[2], t)

    return V(dx⁺..., dxʳ...)
end
function noisejoint(jointx::V, p, t) where V
    x⁺ = @view jointx[1:3]
    xʳ = @view jointx[4:6]

    Σ⁺ = F(SVector{3}(x⁺), p[1], t)
    Σʳ = F(SVector{3}(xʳ), p[2], t)

    SMatrix{6, 3}(
        Σ⁺[1], 0, 0, Σʳ[1], 0, 0,
        0, Σ⁺[2], 0, 0, Σʳ[2], 0,
        0, 0, Σ⁺[3], 0, 0, Σʳ[3]
    )
end

counterfactualsdefn = SDEFunction(Fjoint, noisejoint)
counterfactualprob = SDEProblem(counterfactualsdefn, jointx₀, (0., 100.), counterfactualregretparams; noise_rate_prototype = SMatrix{6, 3}(zeros(6*3)))

counterfactual = solve(counterfactualprob)

function premium(counterfactual, (Hitp, Hʳitp), model)
    P = Vector{Float64}(undef, length(counterfactual))

    for (i, t) in enumerate(counterfactual.t)
        T, m, y, Tʳ, mʳ, yʳ = counterfactual(t)
    
        ∂ₘH = ForwardDiff.derivative(m -> Hitp(T, m, t), m)
        Y = exp(y) * model.economy.Y₀
        M = exp(m) * model.climate.hogg.Mᵖ
        s = scc(∂ₘH, Y, M, model)
    
        ∂ₘHʳ = ForwardDiff.derivative(m -> Hʳitp(Tʳ, m, t), mʳ)
        Yʳ = exp(yʳ) * model.economy.Y₀
        Mʳ = exp(mʳ) * model.climate.hogg.Mᵖ
        sʳ = scc(∂ₘHʳ, Yʳ, Mʳ, model)

        P[i] = (s - sʳ) / s
    end

    return P
end

counterfactualensemble = solve(EnsembleProblem(counterfactualprob); trajectories = 1_000)
P = [premium(sim, (Hitp, Hʳitp), model) for sim in counterfactualensemble];

## Premium trajectories
let
    yearlytime = 0:Int(horizon)

    # Interpolate each premium trajectory onto a regular yearly grid
    Pgrid = Matrix{Float64}(undef, length(yearlytime), length(P))
    for (j, (Pj, sim)) in enumerate(zip(P, counterfactualensemble))
        pitp = linear_interpolation(sim.t, Pj; extrapolation_bc = Flat())
        Pgrid[:, j] = pitp.(yearlytime)
    end

    Pquantiles = [quantile(Pgrid[i, :], (0.05, 0.5, 0.95)) for i in 1:size(Pgrid, 1)]

    medianopts = @pgf {line_width = LINE_WIDTH}
    confidenceopts = @pgf {draw = "none", forget_plot}
    fillopts = @pgf {fill = "black", opacity = 0.15, forget_plot}

    Pmedian = @pgf Plot({medianopts..., color = "black"},
                        Coordinates(yearlytime, getindex.(Pquantiles, 2)))
    Plower  = @pgf Plot({confidenceopts..., name_path = "Plow"},
                        Coordinates(yearlytime, getindex.(Pquantiles, 1)))
    Pupper  = @pgf Plot({confidenceopts..., name_path = "Phigh"},
                        Coordinates(yearlytime, getindex.(Pquantiles, 3)))
    Pfill   = @pgf Plot(fillopts, raw"fill between [of=Plow and Phigh]")

    ytick = 0:0.2:1.
    yticklabels = [@sprintf("\\footnotesize %.0f\\%%", 100y) for y in ytick]
    yearticks = 0:20:horizon

    premiumfig = @pgf Axis({
            width = raw"0.98\linewidth", height = raw"0.35\linewidth",
            grid = "both",
            xmin = 0, xmax = horizon,
            ymin = 0., ymax = 1.,
            xtick = yearticks,
            xticklabels = 2020 .+ Int.(yearticks),
            xticklabel_style = {rotate = 45},
            xlabel = "Year",
            ytick = ytick,
            yticklabels = yticklabels,
            ylabel = L"Premium $P_t$",
        }, Pmedian, Plower, Pupper, Pfill)

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "regret-simfig-premium.tikz"), premiumfig; include_preamble = true)
    end

    premiumfig
end

