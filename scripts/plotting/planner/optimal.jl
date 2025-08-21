using Revise
using JLD2, UnPack, DataStructures
using FastClosures

using StaticArrays
using Statistics
using Model, Grid
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations
using Dierckx

using Random;
Random.seed!(11148705);

using Plots, PGFPlotsX
using LaTeXStrings, Printf
using Colors, ColorSchemes

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}", raw"\usetikzlibrary{patterns}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{siunitx}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\DeclareSIUnit{\ppm}{p.p.m.}")

includet("../../../src/valuefunction.jl")
includet("../../../src/extensions.jl")
includet("../utils.jl")
includet("../../utils/saving.jl")
includet("../../utils/simulating.jl")

damage = Kalkuhl;
withnegative = false
abatementtype = withnegative ? "negative" : "constrained"
DATAPATH = "data/simulation-large-lbfgs";

SAVEFIG = true;
plotpath = joinpath("papers/job-market-paper/submission/plots", abatementtype)
if !isdir(plotpath)
    mkpath(plotpath)
end

horizon = 20.
tspan = (0., horizon)

begin # Import results and interpolations
    simulationfiles = listfiles(DATAPATH)
    nfiles = length(simulationfiles)
    G = loadproblem(first(simulationfiles)) |> last
    models = AbstractModel[]
    interpolations = Dict{AbstractModel,NTuple{2,Base.Callable}}()

    for (i, filepath) in enumerate(simulationfiles)
        model, _ = loadproblem(filepath)
        abatementdir = splitpath(filepath)[3]

        if (model.damages isa damage) && (abatementtype == abatementdir)
            print("Loading file $i / $(nfiles): $(filepath)\r")

            states = loadtotal(filepath; tspan=tspan) |> first
            interpolations[model] = buildinterpolations(states, G)
            push!(models, model)
        end
    end
end;

begin
    calibrationfilepath = "data/calibration.jld2"
    @assert isfile(calibrationfilepath)

    calibrationfile = jldopen(calibrationfilepath, "r+")
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = calibrationfile
    close(calibrationfile)
end

begin # Plot estetics
    tippingmodels = filter(model -> model isa TippingModel, models)
    extremamodel = (models[2], models[1])

    PALETTE = colorschemes[:grays]
    colors = get(PALETTE, range(0, 0.6; length=length(extremamodel)))

    TEMPLABEL = raw"Temperature deviations $T_t \; [\si{\degree\Celsius}]$"
    LINE_WIDTH = 2.5
    hogg = Hogg()

    Tspace = range(G.domains[1]...; length=size(G, 1))
    mspace = range(G.domains[2]...; length=size(G, 2))

    yearlytime = range(tspan[1], tspan[2]; step=1.)

    ΔTmin = Tspace[1] - hogg.Tᵖ
    ΔTmax = Tspace[end] - hogg.Tᵖ
    temperatureticks = makedeviationtickz(ΔTmin, ΔTmax, hogg; step=1, digits=0)
end;

# Policies
if false # FIXME: Redo.
    policyfig = @pgf GroupPlot({group_style = {group_size = "2 by 1", horizontal_sep = "1em"}})

    for (k, model) in enumerate(extremamodel)
        _, policyitp = interpolations[model]

        stableabatement = @closure m -> begin
            T = only(Tstable(m, model))
            state = wrap(T, m, model.hogg)
            return policyitp(0., state).ε
        end

        abatements = map(stableabatement, mspace)

        left = findfirst(a -> length(a) > 1, abatements)
        right = findlast(a -> length(a) > 1, abatements)

        mtick = collect(range(extrema(mspace)...; length=5))[2:(end-1)]
        mticklabels = ["$l" for l in @. Int(round(exp(mtick)))]

        push!(mtick, log(model.hogg.M₀))
        push!(mticklabels, L"M_0")

        idxs = sortperm(mtick)

        mtick = mtick[idxs]
        mticklabels = mticklabels[idxs]

        leftopts = @pgf i > 1 ? {
            yticklabels = raw"\empty"
        } : {
            ylabel = L"Abated fraction $\varepsilon_t$",
            xlabel = L"Carbon Concentration $M_t \; [\si{\ppm}]$ ",
            x_label_style = {anchor = "north west"},
        }

        policyax = @pgf Axis({
            set_layers, mark_layer = "axis background",
            title = labelsbymodel[model],
            xmin = minimum(mspace), xmax = maximum(mspace),
            enlarge_x_limits = 0.01,
            xtick = mtick, xticklabels = mticklabels,
            grid = "both", ymin = 0, ymax = 1,
            enlarge_y_limits = 0.01,
            width = raw"0.5\linewidth", height = raw"0.5\linewidth", leftopts...
        })



        lowerplot = @pgf Plot(
            {line_width = LINE_WIDTH, color = colors[2]},
            Coordinates(mspace[1:right], lowerpolicy)
        )

        lowermarker = @pgf Plot(
            {only_marks, color = colors[2], forget_plot},
            Coordinates(mspace[[right]], lowerpolicy[[end]])
        )

        push!(policyax, lowerplot, lowermarker)

        if i > 1
            push!(policyax, LegendEntry(raw"\footnotesize Low $T$"))
        end

        upperplot = @pgf Plot(
            {line_width = LINE_WIDTH, color = colors[1]},
            Coordinates(mspace[left:end], upperpolicy)
        )

        uppermarker = @pgf Plot(
            {only_marks, forget_plot, color = colors[1]},
            Coordinates(mspace[[left]], upperpolicy[[1]])
        )

        push!(policyax, upperplot, uppermarker)

        if i > 1
            push!(policyax, LegendEntry(raw"\footnotesize High $T$"))
        end

        push!(policyfig, policyax)
    end


    @pgf policyfig["legend style"] = raw"at = {(0.95, 0.3)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, abatementtype, "policyslice.tikz"), policyfig; include_preamble=true)
    end

    policyfig
end

# -- Make simulation of optimal trajectories
begin
    TRAJECTORIES = 10_000
    economy = Economy()
    m₀ = log(hogg.M₀ / hogg.Mᵖ)
    X₀ = SVector(hogg.T₀, m₀, 1.)
    u₀ = SVector(X₀..., 0., 0., 0.) # Introduce three 0s for costs

    simulations = Dict{AbstractModel,EnsembleSolution}()
    for (i, model) in enumerate(models)
        _, policyitp = interpolations[model]

        parameters = (model, calibration, policyitp)

        problem = SDEProblem(F, noise, u₀, tspan, parameters)
        ensembleprob = EnsembleProblem(problem)

        simulation = solve(ensembleprob, SOSRI(); trajectories=TRAJECTORIES, saveat=1.)
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

    qs = [0.01, 0.5, 0.99]

    temperatureticks = makedeviationtickz(0., 6., hogg; step=1, digits=0)


    for (k, model) in enumerate(extremamodel)
        simulation = simulations[model]
        _, policyitp = interpolations[model]
        ts = simulation[1].t

        εfn = @closure (t, u) -> begin
            state = wrap(u[1], u[2], hogg)
            return policyitp(t, state).ε
        end

        abatement = computeonsim(simulation, εfn)
        abatementquantiles = timequantiles(abatement, qs)

        for q in eachcol(abatementquantiles)
            smooth!(q, 5)
        end

        εmedianplot = @pgf Plot(medianopts, Coordinates(ts, abatementquantiles[:, 2]))
        εlower = @pgf Plot({confidenceopts..., name_path = "lower"}, Coordinates(ts, abatementquantiles[:, 1]))
        εupper = @pgf Plot({confidenceopts..., name_path = "upper"}, Coordinates(ts, abatementquantiles[:, 3]))

        εfill = @pgf Plot(fillopts, raw"fill between [of=lower and upper]")

        Eoptionfirst = @pgf k > 1 ? {yticklabel = raw"\empty"} : {ylabel = L"Abated fraction $\varepsilon_t$"}

        @pgf push!(simfig, {figopts...,
                xticklabel = raw"\empty",
                ymin = 0, ymax = abatementtype == "negative" ? 2. : 1.,
                scaled_y_ticks = false, title = labelofmodel(model), Eoptionfirst...,
            }, εmedianplot, εlower, εupper, εfill)
    end

    # Makes the temperature plots in the second row
    for (k, model) in enumerate(extremamodel)
        simulation = simulations[model]
        _, policyitp = interpolations[model]
        ts = simulation[1].t

        paths = EnsembleAnalysis.timeseries_point_quantile(simulation, qs, ts)
        Tpaths = first.(paths.u)

        Tmedianplot = @pgf Plot(medianopts, Coordinates(yearlytime, getindex.(Tpaths, 2)))
        Tlowerplot = @pgf Plot({confidenceopts..., name_path = "lower"}, Coordinates(yearlytime, getindex.(Tpaths, 1)))
        Tupperplot = @pgf Plot({confidenceopts..., name_path = "upper"}, Coordinates(yearlytime, getindex.(Tpaths, 3)))

        Tfill = @pgf Plot(fillopts, raw"fill between [of=lower and upper]")

        figticks = yearticks[1:(k > 1 ? end : end - 1)]

        Toptionfirst = @pgf k > 1 ? {
            yticklabel = raw"\empty"
        } : {
            ytick = temperatureticks[1], yticklabels = temperatureticks[2],
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