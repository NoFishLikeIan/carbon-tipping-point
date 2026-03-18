using Revise
using JLD2, UnPack, DataStructures
using FastClosures
using Base.Threads
using SciMLBase
using Statistics
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations, ForwardDiff, FastChebInterp
using StaticArrays

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

damagetype = BurkeHsiangMiguel;
withnegative = true
abatementtype = withnegative ? "negative" : "constrained"
DATAPATH = "data/simulation"; @assert isdir(DATAPATH)

SAVEFIG = true;
PLOTPATH = "../job-market-paper/jeem/plots"
plotpath = joinpath(PLOTPATH, abatementtype)
if !isdir(plotpath) mkpath(plotpath) end

horizon = 100.
tspan = (0., horizon)

## Read available files
begin
    simulationfiles = listfiles(DATAPATH)
    nfiles = length(simulationfiles)
    G = simulationfiles |> first |> loadproblem |> last

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
end;

## Import available simulations
begin
    models = IAM[]
    valuefunctions = Dict{IAM, OrderedDict{Float64, ValueFunction}}()
    interpolations = Dict{IAM, NTuple{2, Interpolations.Extrapolation}}()
    
    for (i, filepath) = enumerate(modelfiles)
        print("Loading $i / $(length(modelfiles))\r")
        values, model, G = loadtotal(filepath; tspan=(0, 1.01horizon))
        interpolations[model] = buildinterpolations(values, G);
        valuefunctions[model] = values;
        push!(models, model)
    end

    sort!(by = m -> m.climate, models, rev = true)
end

## Load calibration
begin
    calibrationpath = "data/calibration"

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
end


## Plot estetics
begin
    extremamodels = (models[1], models[end])
    extremalabels = ("Linear", "Tipping")
    PALETTE = colorschemes[:grays]
    colors = reverse(get(PALETTE, range(0, 0.6; length=length(extremamodels))))

    TEMPLABEL = raw"Temperature deviations $T_t \; [\si{\degree\Celsius}]$"
    LINE_WIDTH = 2.5

    Tspace, mspace = G.ranges

    yearlytime = range(tspan[1], tspan[2]; step=1.)
    T₀ = hogg.T₀
    m₀ = log(hogg.M₀ / hogg.Mᵖ)
    x₀ = Point(T₀, m₀)

    temperatureticks = makedeviationtickz(Tspace[1], Tspace[end]; step=1, digits=2)

    X₀ = SVector(T₀, m₀, 0.)
    u₀ = SVector(X₀..., 0., 0., 0.) # Introduce three 0s for costs
end;

## Initial policies
begin
    mnpprob = ODEProblem((_, calibration, t) -> γ(t, calibration), m₀, (0, horizon), calibration)
    mnp = solve(mnpprob, Tsit5())

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
    yticklabels = [ @sprintf("\\footnotesize %.0f\\%%", 100y) for y in ytick ]  

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
    });

    # Add gray band for ε > 1 (negative emissions)
    bandx = [mmin, mmax]
    bandy = [1.0, 1.45]
    bandcoords = vcat([(x, bandy[1]) for x in bandx], [(x, bandy[2]) for x in reverse(bandx)])
    bandpoly = @pgf Plot({fill = "gray", opacity = 0.2, draw = "none", forget_plot}, Coordinates(bandcoords))
    push!(policyfig, bandpoly)
    
    timepoints = 0:0.1:horizon
    for (i, model) in enumerate(extremamodels)
        _, α = interpolations[model]
        T̄ = [Tstable(mnp(t), model.climate) for t in timepoints]

        lastlowdx = findlast(Tₜ -> length(Tₜ) > 1, T̄)
        T̄low, tlow = if !isnothing(lastlowdx)
            first.(T̄[1:lastlowdx]), timepoints[1:lastlowdx]
        else
            first.(T̄[1:end]), timepoints[1:end]
        end
        
        firsthighdx = findfirst(Tₜ -> length(Tₜ) > 1, T̄)
        T̄high, thigh = if !isnothing(firsthighdx)
            last.(T̄[firsthighdx:end]), timepoints[firsthighdx:end]
        else
            Float64[], Float64[]
        end
 
        mlow = [mnp(t) for t in tlow]
        mhigh = [mnp(t) for t in thigh]

        εₜlow = [ ε(t, Point(T, m), α(T, m, t), model, calibration) for (T, m, t) in zip(T̄low, mlow, tlow)]
        
        lowcurve = @pgf Plot({ line_width = LINE_WIDTH, color = colors[i], solid }, Coordinates(mlow, εₜlow))
        @pgf push!(policyfig, lowcurve, LegendEntry("\\footnotesize $(extremalabels[i])"))

        if !isempty(T̄high)
            εₜhigh = [ ε(t, Point(T, m), α(T, m, t), model, calibration) for (T, m, t) in zip(T̄high, mhigh, thigh)]
            highcurve = @pgf Plot({ line_width = LINE_WIDTH, color = colors[i], solid, forget_plot }, Coordinates(mhigh, εₜhigh))
            tippingmarker = @pgf Plot({
                mark_options = {fill = colors[i]}, only_marks, forget_plot
            }, Coordinates(mhigh[[1]], εₜhigh[[1]]))
            push!(policyfig, highcurve, tippingmarker)
        end
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "policyfig.tikz"), policyfig; include_preamble=true)
    end

    policyfig
end

## Make simulation of optimal trajectories
begin
    simulations = Dict{IAM,EnsembleSolution}()

    for (i, model) in enumerate(extremamodels)
        αitp = interpolations[model] |> last;
        parameters = (model, calibration, αitp);

        problem = SDEProblem(F, noise, u₀, tspan, parameters)
        ensembleprob = EnsembleProblem(problem)

        simulation = solve(ensembleprob; trajectories = 10_000)
        println("Done with simulation of $i / $(length(extremamodels))\n$model\n")

        simulations[model] = simulation
    end
end

## Optimal abatement figure
begin
    yearticks = 0:20:horizon

    medianopts = @pgf {line_width = LINE_WIDTH}
    confidenceopts = @pgf {draw = "none", forget_plot}
    fillopts = @pgf {fill = "gray", opacity = 0.5, forget_plot}
    
    εtick = 0.4:0.2:1.2
    εticklabels = [ @sprintf("\\footnotesize %.0f\\%%", 100y) for y in εtick ]

    optabatementfig = @pgf Axis({
        width = raw"0.8\textwidth",
        height = raw"0.45\textwidth",
        grid = "both",
        xmin = 0,
        xmax = horizon,
        ymin = 0.35,
        ymax = maximum(εtick),
        xtick = yearticks,
        xticklabels = 2020 .+ Int.(yearticks),
        xticklabel_style = {rotate = 45},
        xlabel = "Year",
        ylabel = raw"\footnotesize Optimal fraction of abated emissions $\varepsilon_t$",
        ytick = εtick,
        yticklabels = εticklabels,
        legend_pos = "north west"
    })

    qs = (0.05, 0.5, 0.95)

    # Add gray band for ε > 1 (negative emissions)
    bandx = (0, horizon)
    bandy = [1.0, maximum(εtick)]
    bandcoords = vcat([(x, bandy[1]) for x in bandx], [(x, bandy[2]) for x in reverse(bandx)])
    bandpoly = @pgf Plot({fill = "gray", opacity = 0.25, draw = "none", forget_plot}, Coordinates(bandcoords))

    push!(optabatementfig, bandpoly)

    # Overlay both models in a single axis
    for (k, model) in enumerate(extremamodels)
        ensemble = simulations[model]
        _, αitp = interpolations[model]

        paths = EnsembleAnalysis.timeseries_point_quantile(ensemble, qs, 0:horizon)

        epaths = NTuple{3, Float64}[]

        for (t, u) in zip(paths.t, paths.u)
            T, m = u[1:2]
            εₜ = ntuple(i -> ε(t, Point(T[i], m[i]), αitp(T[i], m[i], t), model, calibration), 3)

            push!(epaths, εₜ)
        end

        medianpath = getindex.(epaths, 2)
        fullabatementdx = searchsortedfirst(medianpath, 1.)
        fullabatementyear = (0:horizon)[fullabatementdx]

        lowerpath = "elower$(k)"
        upperpath = "eupper$(k)"

        emedianplot = @pgf Plot({medianopts..., color = colors[k],}, Coordinates(0:horizon, medianpath))
        elowerplot = @pgf Plot({confidenceopts..., color = colors[k], name_path = lowerpath}, Coordinates(0:horizon, getindex.(epaths, 1)))
        eupperplot = @pgf Plot({confidenceopts...,  color = colors[k], name_path = upperpath}, Coordinates(0:horizon, getindex.(epaths, 3)))

        efill = @pgf Plot(fillopts, "fill between [of=$lowerpath and $upperpath]")

        fullabatementline = @pgf Plot({ line_width = 1.5, dashed, color = colors[k], forget_plot }, Coordinates([fullabatementyear, fullabatementyear], [0., 1.]))

        fullabatementscatter = @pgf Plot({ only_marks, color = colors[k], forget_plot }, Coordinates([fullabatementyear], [1.]))

        @pgf push!(optabatementfig,
            emedianplot,
            LegendEntry("\\footnotesize $(extremalabels[k])"),
            elowerplot,
            eupperplot,
            efill,
            fullabatementline,
            fullabatementscatter
        )
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "optabatementfig.tikz"), optabatementfig; include_preamble=true)
    end

    optabatementfig
end

## Optimal variable path figure
begin 
    simfig = @pgf GroupPlot({
        group_style = {
            group_size = "$(length(extremamodels)) by 2",
            horizontal_sep = raw"1em"
    }})

    yearticks = 0:20:horizon

    medianopts = @pgf {line_width = LINE_WIDTH}
    confidenceopts = @pgf {forget_plot, line_width = 0.6}
    fillopts = @pgf { opacity = 0.15 }
    figopts = @pgf {width = raw"0.5\textwidth", height = raw"0.35\textwidth", grid = "both", xmin = 0, xmax = horizon}

    qs = (0.1, 0.5, 0.9)
    temperatureticks = makedeviationtickz(1, 3; step=0.5, digits=1)
    mnppath = mnp(0:horizon)
    Mnppath = @. exp(mnppath) * hogg.Mᵖ

    # Carbon concentration in first row
    for (k, model) in enumerate(extremamodels) 
        ensemble = simulations[model]
        paths = EnsembleAnalysis.timeseries_point_quantile(ensemble, qs, 0:horizon)
        mpaths = getindex.(paths.u, 2)
        Mpaths = [@. hogg.Mᵖ * exp(m) for m in mpaths]

        Mmedianplot = @pgf Plot({medianopts..., color = colors[k]}, Coordinates(0:horizon, getindex.(Mpaths, 2)))
        Mlowerplot = @pgf Plot({confidenceopts..., color = colors[k], name_path = "Mlower"}, Coordinates(0:horizon, getindex.(Mpaths, 1)))
        Mupperplot = @pgf Plot({confidenceopts..., color = colors[k], name_path = "Mupper"}, Coordinates(0:horizon, getindex.(Mpaths, 3)))

        Mfill = @pgf Plot(fillopts, raw"fill between [of=Mlower and Mupper]")

        labeloption = @pgf k > 1 ? { yticklabel = raw"\empty" } : { ylabel = L"`Conecntration $M_t \; [\si{ppm}]$" }

        nppath = @pgf Plot({ dashed, color = colors[k], line_width = LINE_WIDTH }, Coordinates(0:horizon, Mnppath.u))

        @pgf push!(simfig, {figopts...,
            xticklabel = raw"\empty", ymin = hogg.M₀, ymax = 600.,
            title = extremalabels[k], labeloption...,
        }, Mmedianplot, Mlowerplot, Mupperplot, Mfill, nppath)
    end

    # Temperature in second row
    for (k, model) in enumerate(extremamodels)
        ensemble = simulations[model]

        paths = EnsembleAnalysis.timeseries_point_quantile(ensemble, qs, 0:horizon)
        Tpaths = first.(paths.u)

        Tmedianplot = @pgf Plot({medianopts..., color = colors[k]}, Coordinates(0:horizon, getindex.(Tpaths, 2)))
        Tlowerplot = @pgf Plot({confidenceopts..., color = colors[k], name_path = "Tlower"}, Coordinates(0:horizon, getindex.(Tpaths, 1)))
        Tupperplot = @pgf Plot({confidenceopts..., color = colors[k], name_path = "Tupper"}, Coordinates(0:horizon, getindex.(Tpaths, 3)))

        Tfill = @pgf Plot(fillopts, raw"fill between [of=Tlower and Tupper]")

        Tnppath = [Tstable(m, model.climate) |> first for m in mnppath.u]
        nppath = @pgf Plot({ dashed, color = "gray", line_width = LINE_WIDTH }, Coordinates(0:horizon, Tnppath))

        figticks = yearticks[1:(k > 1 ? end : end - 1)]

        Toptionfirst = @pgf k > 1 ? {
            yticklabel = raw"\empty"
        } : {
            ytick = temperatureticks[1], 
            yticklabels = temperatureticks[2],
            ylabel = raw"Temperature $T_t$"
        }

        @pgf push!(simfig, {figopts...,
                ymin = minimum(temperatureticks[1]),
                ymax = maximum(temperatureticks[1]),
                xtick = figticks,
                xticklabels = 2020 .+ Int.(figticks),
                xticklabel_style = {rotate = 45},
                xlabel = "Year",
                Toptionfirst...
            }, Tmedianplot, Tlowerplot, Tupperplot, Tfill, nppath)
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "simfig.tikz"), simfig; include_preamble=true)
    end

    simfig
end


## Cost figure
if false
    costfig = @pgf GroupPlot({
        group_style = {
            group_size = "$(length(simulations)) by 1",
            horizontal_sep = raw"2em"
        }})

    decadetime = 0:10:horizon
    decadeslabels = ["$(2020 + dec)s" for dec in Int.(decadetime[1:end-1])]

    patterns = [nothing, "crosshatch", "horizontal lines"]
    labels = ["Climate Damages", "Abatement", "Adjustment costs"]
    idxs = [6, 4, 5]

    ytick = 0:0.005:0.02
    yticklabels = [L"%$(y * 100)\%" for y in ytick]


    figopts = @pgf {
        width = raw"0.5\textwidth", height = raw"0.5\textwidth", grid = "both",
        symbolic_x_coords = decadeslabels,
        xticklabel_style = {rotate = 45, align = "right"}, xtick = "data",
        enlarge_x_limits = 0.1,
        ymin = 0, ymax = maximum(ytick),
        ybar_stacked, bar_width = "2ex",
        x = "2.5ex", ytick = ytick, yticklabels = yticklabels, reverse_legend,
        scaled_y_ticks = false,
        legend_style = {at = "{(1.1, 0.8)}", anchor = "east"},
        ticklabel_style = {font=L"\footnotesize"}
    }

    for (k, model) in enumerate(extremamodels)
        sim = simulations[model]

        decadespath = EnsembleAnalysis.timeseries_point_median(sim, decadetime)
        decadechange = diff(decadespath.u) / step(decadetime) # average

        kopts = @pgf k > 1 ? {
            yticklabels = raw"\empty"
        } : {
            ylabel = L"\small Average decade costs, \% $Y_t$"
        }

        barchart = @pgf Axis({
            figopts..., kopts...,
            title = extremalabels[k]
        })

        for i in 1:3
            pattern = patterns[i]
            label = labels[i]
            idx = idxs[i]

            coords = Coordinates(decadeslabels, getindex.(decadechange, idx))

            bar = @pgf Plot({
                    ybar,
                    pattern = pattern
                }, coords)

            push!(barchart, bar)

            if k > 1
                push!(barchart, LegendEntry("\\footnotesize $label"))
            end
        end

        push!(costfig, barchart)
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "opt-costs.tikz"), costfig; include_preamble=true)
    end

    costfig
end