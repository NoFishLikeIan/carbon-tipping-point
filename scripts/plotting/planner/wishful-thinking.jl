using Revise
using Suppressor: @suppress
using JLD2, UnPack
using FastClosures

using Random

using Plots, PGFPlotsX
using LaTeXStrings, Printf
using Colors, ColorSchemes

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}",raw"\usetikzlibrary{patterns}")

using Statistics
using Model, Grid
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations: Extrapolation
using Dierckx, ImageFiltering

includet("../utils.jl")
includet("../../utils/saving.jl")
includet("../../utils/simulating.jl")

SAVEFIG = false;
ALLOWNEGATIVE = false;
datapath = "data/simulation-large";
experimentpath = "data/experiments/simulation-large";
PLOT_HORIZON = 80.

begin # Default parameters
    θ = 10.
    ψ = 0.75
    damages = GrowthDamages

    ismodel = @closure model -> begin
        model.preferences.θ == θ &&
        model.preferences.ψ == ψ &&
        model.damages isa damages
    end

    plotpath = joinpath("plots", damages == LevelDamages ? "damage-robust" : "")
    calibration = load_object("data/calibration.jld2")
end

begin # Import results and interpolations
    simulationfilespath = joinpath(datapath, ALLOWNEGATIVE ? "negative" : "constrained")

    simulationfiles = listfiles(simulationfilespath)

    models = AbstractModel[]
    interpolations = Dict{AbstractModel, Dict{Symbol, Extrapolation}}();

    for filepath in simulationfiles
        result = loadtotal(filepath)
        model = last(result)

        if ismodel(model)
            println("Loading $(filepath)")
            push!(models, model)
            interpolations[model] = buildinterpolations(result)
        end
    end
end


begin # Plot estetics
    PALETTE = colorschemes[:grays];
    colors = get(PALETTE, [0., 0.5]);
    
    TEMPLABEL = L"Temperature deviations $T_t$"
    LINE_WIDTH = 2.5

    ΔTmin = Hogg().T₀ - Hogg().Tᵖ
    ΔTmax = 3. 

    ΔTspace = range(ΔTmin, ΔTmax; length = 201);
    Tspace = ΔTspace .+ Hogg().Tᵖ
    Tmin, Tmax = extrema(Tspace)

    model = first(models)
    nofeedback = Albedo(0., 0., model.albedo.λ₁, 0)
    nofeedbackmodel = TippingModel(nofeedback, model.hogg, model.preferences, model.damages, model.economy)

    mspace = range(mstable(Tmin, nofeedbackmodel), mstable(Tmax, nofeedbackmodel); length = length(ΔTspace))

    yearlytime = range(0., PLOT_HORIZON; step = 1.)
    temperatureticks = makedeviationtickz(ΔTmin, ΔTmax, model; step = 1, digits = 0)
end;

begin
    tippingmodel = filter(m -> m isa TippingModel, models)
    
    imminentmodel = models[findmin(m -> m.albedo.Tᶜ, tippingmodel) |> last]
    remotemodel = models[findmax(m -> m.albedo.Tᶜ, tippingmodel) |> last]

    X₀ = [imminentmodel.hogg.T₀, log(imminentmodel.hogg.M₀), log(imminentmodel.economy.Y₀)];

    u₀ = [X₀..., 0., 0., 0.] # Introduce three 0s for costs

    initpolicies = (interpolations[remotemodel][:χ], interpolations[remotemodel][:α]);

    parameters = (imminentmodel, initpolicies, calibration)

    wtprob = SDEProblem(Fbreakdown!, Gbreakdown!, u₀, (0., 200.), parameters);

    function discovery(u, t, integrator)
        model = integrator.p[1]

        return (model.albedo.Tᶜ - u[1])
    end

    function updatepolicy!(integrator)
        return
        integrator.p[2] = (interpolations[imminentmodel][:χ], interpolations[imminentmodel][:α])
    end

    cb = ContinuousCallback(discovery, updatepolicy!)
end

begin # Simulation
    TRAJECTORIES = 10_000

    wsim = solve(EnsembleProblem(wtprob); trajectories = TRAJECTORIES, callback = cb)
end;

begin
    simfig = @pgf GroupPlot({
        group_style = {
            group_size = "2 by 1",
            horizontal_sep = raw"5em"
        }});

    SMOOTH_FACTOR = 5
    yearticks = 0:20:PLOT_HORIZON

    medianopts = @pgf { line_width = LINE_WIDTH }
    confidenceopts = @pgf { draw = "none", forget_plot }
    fillopts = @pgf { fill = "gray", opacity = 0.5 }
    figopts = @pgf { width = raw"0.45\textwidth", height = raw"0.35\textwidth", grid = "both", xmin = 0, xmax = PLOT_HORIZON }

    qs = [0.05, 0.5, 0.95]

    temperatureticks = makedeviationtickz(0., 6., imminentmodel; step = 1, digits = 0)

    Mticks = 400:40:520

    begin # εₜ
        Efn = (T, m, y, _, _, _, t) -> begin
            αitp = interpolations[remotemodel][:α]

            abatement = αitp(T, m, t)
            return ε(t, exp(m), abatement, imminentmodel, calibration)
        end

        EM = computeonsim(wsim, Efn, yearlytime);

        Equantiles = timequantiles(EM, qs);
        smoothquantile!.(eachcol(Equantiles), SMOOTH_FACTOR)

        Emedianplot = @pgf Plot(medianopts, Coordinates(yearlytime, Equantiles[:, 2]))
        Elowerplot = @pgf Plot({ confidenceopts..., name_path = "lower" }, Coordinates(yearlytime, Equantiles[:, 1]))
        Eupperplot = @pgf Plot({ confidenceopts..., name_path = "upper" }, Coordinates(yearlytime, Equantiles[:, 3]))

        Efill = @pgf Plot(fillopts, raw"fill between [of=lower and upper]")


        @pgf push!(simfig, {figopts...,
            ymin = 0, ymax = 1,
            xtick = yearticks,
            xticklabels = 2020 .+ Int.(yearticks),
            xticklabel_style = { rotate = 45 },
            scaled_y_ticks = false, ylabel = L"\footnotesize Abated fraction $\varepsilon_t$"
        }, Emedianplot, Elowerplot, Eupperplot, Efill)
    end

    begin # Tₜ
        paths = EnsembleAnalysis.timeseries_point_quantile(wsim, qs, yearlytime)
        Tpaths = first.(paths.u)

        Tmedianplot = @pgf Plot(medianopts, Coordinates(yearlytime, getindex.(Tpaths, 2)))
        Tlowerplot = @pgf Plot({ confidenceopts..., name_path = "lower" }, Coordinates(yearlytime, getindex.(Tpaths, 1)));
        Tupperplot = @pgf Plot({ confidenceopts..., name_path = "upper" }, Coordinates(yearlytime, getindex.(Tpaths, 3)));

        Tfill = @pgf Plot(fillopts, raw"fill between [of=lower and upper]")

        @pgf push!(simfig, {figopts...,
            ymin = minimum(temperatureticks[1]),
            ymax = maximum(temperatureticks[1]),
            xtick = yearticks,
            xticklabels = 2020 .+ Int.(yearticks),
            xticklabel_style = { rotate = 45 },
            ytick = temperatureticks[1], yticklabels = temperatureticks[2],
            ylabel = raw"\footnotesize Temperature $T_t$"
        }, Tmedianplot, Tlowerplot, Tupperplot, Tfill)
    end;


    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "wf-simfig.tikz"), simfig; include_preamble = true)
    end

    simfig
end

# Cost breakdown
begin
    decadetime = 0:10:PLOT_HORIZON
    decadeslabels = ["$(2020 + dec)s" for  dec in Int.(decadetime[1:end - 1])]

    patterns = [nothing, "crosshatch", "horizontal lines"]
    labels = ["Climate Damages", "Abatement", "Adjustment costs"]
    idxs = [6, 5, 4]

    ytick = 0:0.025:0.1
    yticklabels = [L"%$(y * 100)\%" for y in ytick]

    decadespath = EnsembleAnalysis.timeseries_point_median(wsim, decadetime)
    decadechange = diff(decadespath.u) / step(decadetime) # average

    barchart = @pgf Axis({
        width = raw"0.7\textwidth", height = raw"0.5\textwidth", grid = "both", 
        symbolic_x_coords = decadeslabels,
        xticklabel_style = { rotate = 45, align = "right" }, xtick = "data",
        enlarge_x_limits = 0.1,
        ymin = 0, ymax = maximum(ytick),
        ybar_stacked, bar_width = "4ex",
        x = "7ex", ytick = ytick, yticklabels = yticklabels, reverse_legend,
        scaled_y_ticks = false,
        ylabel = L"\small Average decade costs, \% $Y_t$"
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
        push!(barchart, LegendEntry("\\footnotesize $label"))
    end

    if SAVEFIG
        PGFPlotsX.pgfsave(joinpath(plotpath, "wf-costs.tikz"), barchart; include_preamble = true)
    end

    barchart
end

# -- Prudence
begin
    imminentpolicies = (interpolations[imminentmodel][:χ], interpolations[imminentmodel][:α]);
    parameters = (remotemodel, imminentpolicies, calibration)

    prudenceprob = SDEProblem(Fbreakdown!, Gbreakdown!, u₀, (0., 200.), parameters)

    prudsim = solve(EnsembleProblem(prudenceprob); trajectories = TRAJECTORIES)
end;

begin
    simfig = @pgf GroupPlot({
        group_style = {
            group_size = "2 by 1",
            horizontal_sep = raw"5em"
        }});

    begin # εₜ
        Efn = (T, m, y, _, _, _, t) -> begin
            αitp = interpolations[imminentmodel][:α]

            abatement = αitp(T, m, t)
            return ε(t, exp(m), abatement, imminentmodel, calibration)
        end

        EM = computeonsim(prudsim, Efn, yearlytime);

        Equantiles = timequantiles(EM, qs);
        smoothquantile!.(eachcol(Equantiles), SMOOTH_FACTOR)

        Emedianplot = @pgf Plot(medianopts, Coordinates(yearlytime, Equantiles[:, 2]))
        Elowerplot = @pgf Plot({ confidenceopts..., name_path = "lower" }, Coordinates(yearlytime, Equantiles[:, 1]))
        Eupperplot = @pgf Plot({ confidenceopts..., name_path = "upper" }, Coordinates(yearlytime, Equantiles[:, 3]))

        Efill = @pgf Plot(fillopts, raw"fill between [of=lower and upper]")


        @pgf push!(simfig, {figopts...,
            ymin = 0, ymax = 1,
            xtick = yearticks,
            xticklabels = 2020 .+ Int.(yearticks),
            xticklabel_style = { rotate = 45 },
            scaled_y_ticks = false, ylabel = L"\footnotesize Abated fraction $\varepsilon_t$"
        }, Emedianplot, Elowerplot, Eupperplot, Efill)
    end

    begin # Tₜ
        paths = EnsembleAnalysis.timeseries_point_quantile(prudsim, qs, yearlytime)
        Tpaths = first.(paths.u)

        Tmedianplot = @pgf Plot(medianopts, Coordinates(yearlytime, getindex.(Tpaths, 2)))
        Tlowerplot = @pgf Plot({ confidenceopts..., name_path = "lower" }, Coordinates(yearlytime, getindex.(Tpaths, 1)));
        Tupperplot = @pgf Plot({ confidenceopts..., name_path = "upper" }, Coordinates(yearlytime, getindex.(Tpaths, 3)));

        Tfill = @pgf Plot(fillopts, raw"fill between [of=lower and upper]")

        @pgf push!(simfig, {figopts...,
            ymin = minimum(temperatureticks[1]),
            ymax = maximum(temperatureticks[1]),
            xtick = yearticks,
            xticklabels = 2020 .+ Int.(yearticks),
            xticklabel_style = { rotate = 45 },
            ytick = temperatureticks[1], yticklabels = temperatureticks[2],
            ylabel = raw"\footnotesize Temperature $T_t$"
        }, Tmedianplot, Tlowerplot, Tupperplot, Tfill)
    end;


    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "p-simfig.tikz"), simfig; include_preamble = true)
    end

    simfig
end

# Cost breakdown
begin
    decadespath = EnsembleAnalysis.timeseries_point_median(prudsim, decadetime)
    decadechange = diff(decadespath.u) / step(decadetime) # average

    barchart = @pgf Axis({
        width = raw"0.7\textwidth", height = raw"0.5\textwidth", grid = "both", 
        symbolic_x_coords = decadeslabels,
        xticklabel_style = { rotate = 45, align = "right" }, xtick = "data",
        enlarge_x_limits = 0.1,
        ymin = 0, ymax = maximum(ytick),
        ybar_stacked, bar_width = "4ex",
        x = "7ex", ytick = ytick, yticklabels = yticklabels, reverse_legend,
        scaled_y_ticks = false,
        ylabel = L"\small Average decade costs, \% $Y_t$"
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
        push!(barchart, LegendEntry("\\footnotesize $label"))
    end

    if SAVEFIG
        PGFPlotsX.pgfsave(joinpath(plotpath, "p-costs.tikz"), barchart; include_preamble = true)
    end

    barchart
end