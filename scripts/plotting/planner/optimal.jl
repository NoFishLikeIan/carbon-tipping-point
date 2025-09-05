using Revise
using JLD2, UnPack, DataStructures
using FastClosures
using Base.Threads
using SciMLBase
using Statistics
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations, Dierckx
using StaticArrays

using Model, Grid

using Random; Random.seed!(11148705);

using Plots, PGFPlotsX, Contour
using LaTeXStrings, Printf
using Colors, ColorSchemes

# pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}", raw"\usetikzlibrary{patterns}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{siunitx}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\DeclareSIUnit{\ppm}{p.p.m.}")

includet("../../../src/extend/model.jl")
includet("../../../src/valuefunction.jl")
includet("../../../src/extend/valuefunction.jl")
includet("../utils.jl")
includet("../../utils/saving.jl")
includet("../../utils/simulating.jl")

damage = Kalkuhl;
withnegative = true
abatementtype = withnegative ? "negative" : "constrained"
DATAPATH = "data/simulation-dense"; @assert isdir(DATAPATH)

SAVEFIG = false;
plotpath = joinpath("papers/job-market-paper/submission/plots", abatementtype)
if !isdir(plotpath) mkpath(plotpath) end

horizon = 80.
tspan = (0., horizon)

begin # Read available files
    simulationfiles = listfiles(DATAPATH)
    nfiles = length(simulationfiles)
    G = simulationfiles |> first |> loadproblem |> last
    
    modelfiles = String[]
    for (i, filepath) in enumerate(simulationfiles)
        print("Reading $i / $(length(simulationfiles))\r")
        model, _ = loadproblem(filepath)
        abatementdir = splitpath(filepath)[end - 1]

        if (model.damages isa damage) && (abatementtype == abatementdir)
            push!(modelfiles, filepath)
        end
    end
end;

begin # Import available files
    models = AbstractModel[]
    valuefunctions = Dict{AbstractModel, OrderedDict{Float64, ValueFunction}}()
    interpolations = Dict{AbstractModel, NTuple{2, Interpolations.Extrapolation}}()
    
    for (i, filepath) = enumerate(modelfiles)
        print("Loading $i / $(length(modelfiles))\r")
        values, model, G = loadtotal(filepath; tspan=(0, 1.2horizon))
        interpolations[model] = buildinterpolations(values, G);
        valuefunctions[model] = values
        push!(models, model)
    end

    sort!(models)
end

begin
    calibrationfilepath = "data/calibration.jld2"
    @assert isfile(calibrationfilepath)

    calibrationfile = jldopen(calibrationfilepath, "r+")
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = calibrationfile
    close(calibrationfile)
end

begin # Plot estetics
    extremamodel = extrema(models)
    PALETTE = colorschemes[:grays]
    colors = get(PALETTE, range(0, 0.6; length=length(extremamodel)))

    TEMPLABEL = raw"Temperature deviations $T_t \; [\si{\degree\Celsius}]$"
    LINE_WIDTH = 2.5

    Tspace, mspace = G.ranges

    yearlytime = range(tspan[1], tspan[2]; step=1.)
    T₀ = hogg.T₀
    m₀ = log(hogg.M₀ / hogg.Mᵖ)
    x₀ = Point(T₀, m₀)

    ΔTmin = Tspace[1] - hogg.Tᵖ
    ΔTmax = Tspace[end] - hogg.Tᵖ
    temperatureticks = makedeviationtickz(ΔTmin, ΔTmax, hogg; step=1, digits=0)

    economy = Economy()

    X₀ = SVector(T₀, m₀, 1.)
    u₀ = SVector(X₀..., 0., 0., 0.) # Introduce three 0s for costs
end;

# Policies
if false t = 0.
    policyfig = @pgf GroupPlot({group_style = {group_size = "2 by 1", horizontal_sep = "1em"}});

    @pgf policyfig["legend style"] = raw"at = {(0.95, 0.3)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, abatementtype, "policyslice.tikz"), policyfig; include_preamble=true)
    end

    policyfig
end;

# -- Make simulation of optimal trajectories
begin
    trajectories = 10_000

    simulations = Dict{AbstractModel,EnsembleSolution}()
    for (i, model) in enumerate(models)
        αitp = interpolations[model] |> last;
        parameters = (model, calibration, αitp);

        problem = SDEProblem(F, noise, u₀, tspan, parameters)
        ensembleprob = EnsembleProblem(problem)

        simulation = solve(ensembleprob, SRA3(); trajectories)
        println("Done with simulation of $i / $(length(models))\n$model")

        simulations[model] = simulation
    end
end

begin
    simfig = @pgf GroupPlot({
    group_style = {
        group_size = "$(length(extremamodel)) by 2",
        horizontal_sep = raw"1em",
    }})

    yearticks = 0:20:horizon

    medianopts = @pgf {line_width = LINE_WIDTH}
    confidenceopts = @pgf {draw = "none", forget_plot}
    fillopts = @pgf {fill = "gray", opacity = 0.5}
    figopts = @pgf {width = raw"0.5\textwidth", height = raw"0.35\textwidth", grid = "both", xmin = 0, xmax = horizon}

    qs = (0.05, 0.5, 0.95)
    temperatureticks = makedeviationtickz(1, 2.5, hogg; step=0.5, digits=1)

    # Carbon concentration in first row
    for (k, model) in enumerate(extremamodel) 
        simulation = simulations[model]
        ts = first(simulation).t

        paths = EnsembleAnalysis.timeseries_point_quantile(simulation, qs, ts)
        
        M = Matrix{Float64}(undef, length(qs), length(paths));
        for (k, row) in enumerate(getindex.(paths.u, 2))
            @. M[:, k] = hogg.Mᵖ * exp(row)
        end

        medianplot = @pgf Plot(medianopts, Coordinates(ts, M[2, :]))
        lowerplot = @pgf Plot({confidenceopts..., name_path = "lower"}, Coordinates(ts, M[3, :]))
        upperplot = @pgf Plot({confidenceopts..., name_path = "upper"}, Coordinates(ts, M[1, :]))
        fill = @pgf Plot(fillopts, raw"fill between [of=lower and upper]")


        labeloption = @pgf k > 1 ? { yticklabel = raw"\empty" } : { ylabel = L"`Conecntration $M_t \; [\si{ppm}]$" }
        @pgf push!(simfig, {figopts...,
                xticklabel = raw"\empty",
                title = labelofmodel(model), labeloption...,
            }, medianplot, lowerplot, upperplot, fill)
    end

    # Temperature in second row
    for (k, model) in enumerate(extremamodel)
        simulation = simulations[model]
        _, policyitp = interpolations[model]
        ts = simulation[1].t

        paths = EnsembleAnalysis.timeseries_point_quantile(simulation, qs, ts)
        Tpaths = first.(paths.u)

        Tmedianplot = @pgf Plot(medianopts, Coordinates(ts, getindex.(Tpaths, 2)))
        Tlowerplot = @pgf Plot({confidenceopts..., name_path = "lower"}, Coordinates(ts, getindex.(Tpaths, 1)))
        Tupperplot = @pgf Plot({confidenceopts..., name_path = "upper"}, Coordinates(ts, getindex.(Tpaths, 3)))

        Tfill = @pgf Plot(fillopts, raw"fill between [of=lower and upper]")

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
            }, Tmedianplot, Tlowerplot, Tupperplot, Tfill)
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "simfig.tikz"), simfig; include_preamble=true)
    end

    simfig
end

begin
    costfig = @pgf GroupPlot({
        group_style = {
            group_size = "$(length(simulations)) by 1",
            horizontal_sep = raw"2em"
        }})

    decadetime = 0:10:horizon
    decadeslabels = ["$(2020 + dec)s" for dec in Int.(decadetime[1:end-1])]

    patterns = [nothing, "crosshatch", "horizontal lines"]
    labels = ["Climate Damages", "Abatement", "Adjustment costs"]
    idxs = [6, 5, 4]

    ytick = 0:0.005:0.03
    yticklabels = [L"%$(y * 100)\%" for y in ytick]

    figopts = @pgf {
        width = raw"0.5\textwidth", height = raw"0.5\textwidth", grid = "both",
        symbolic_x_coords = decadeslabels,
        xticklabel_style = {rotate = 45, align = "right"}, xtick = "data",
        enlarge_x_limits = 0.1,
        ymin = 0, ymax = maximum(ytick),
        ybar_stacked, bar_width = "2.5ex",
        x = "3ex", ytick = ytick, yticklabels = yticklabels, reverse_legend,
        scaled_y_ticks = false,
        legend_style = {at = "{(0.9, 0.9)}", anchor = "west"}
    }

    for (k, model) in enumerate(extremamodel)
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
            title = labelofmodel(model)
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