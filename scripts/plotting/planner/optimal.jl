using Revise
using Suppressor: @suppress
using JLD2, UnPack
using FastClosures

using Random

using Plots, PGFPlotsX
using LaTeXStrings, Printf
using Colors, ColorSchemes

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}", raw"\usetikzlibrary{patterns}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{siunitx}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\DeclareSIUnit{\ppm}{p.p.m.}")

using Statistics
using Model, Grid
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations: Extrapolation
using Dierckx, ImageFiltering

includet("../utils.jl")
includet("../../utils/saving.jl")
includet("../../utils/simulating.jl")

SAVEFIG = true;
ALLOWNEGATIVE = false;
datapath = "data/simulation-large";
PLOT_HORIZON = 80.

begin # Default parameters
    θ = 10.
    ψ = 0.75
    damages = GrowthDamages
    thresholds = [1.5, 2.5]

    ismodel = @closure model -> begin
        model.preferences.θ == θ &&
        model.preferences.ψ == ψ &&
        model.damages isa damages && 
        model.albedo.Tᶜ in thresholds 
    end

    robustpath = if damages == LevelDamages
        "robust/damage"
    elseif maximum(thresholds) > 2.5
        "robust/tipping"
    else
        ""
    end

    plotpath = joinpath("plots", robustpath)
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

begin # Extract relevant models
    tippingmodels = sort(filter(m -> m isa TippingModel, models), by = m -> m.albedo.Tᶜ)[1:2]
    labels = ["Imminent", "Remote"]

    labelsbymodel = Dict{AbstractModel, String}(tippingmodels .=> labels)
    thresholds = unique(m.albedo.Tᶜ for m in tippingmodels)

    calibration = load_object("data/calibration.jld2")
end;

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

    model = first(tippingmodels)
    nofeedback = Albedo(0., 0., model.albedo.λ₁, 0)
    nofeedbackmodel = TippingModel(nofeedback, model.hogg, model.preferences, model.damages, model.economy)

    mspace = range(mstable(Tmin, nofeedbackmodel), mstable(Tmax, nofeedbackmodel); length = length(ΔTspace))

    yearlytime = range(0., PLOT_HORIZON; step = 1.)
    temperatureticks = makedeviationtickz(ΔTmin, ΔTmax, model; step = 1, digits = 0)
end;

# Policies
begin
    policyfig = @pgf GroupPlot({
        group_style = {group_size = "2 by 1", horizontal_sep = "1em"}
    });

    for (i, model) in enumerate(tippingmodels)
        α = interpolations[model][:α];

        function stableabatement(m, t, model)
            steadystates = Model.Tstable(m, model)
            return [ε(t, exp(m), α(T, m, t), model, calibration) for T in steadystates]
        end


        abatements = [stableabatement(m, 40., model) for m in mspace]

        left = findfirst(a -> length(a) > 1, abatements)
        right = findlast(a -> length(a) > 1, abatements)

        ker = ImageFiltering.Kernel.gaussian((10,))
        lowerpolicy = imfilter(first.(abatements[1:right]), ker)
        upperpolicy = imfilter(last.(abatements[left:end]), ker)

        mtick = collect(range(extrema(mspace)...; length = 5))[2:(end - 1)]
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
            x_label_style= { anchor="north west" },
        }

        policyax = @pgf Axis({
            set_layers, mark_layer="axis background",
            title = labelsbymodel[model],
            xmin = minimum(mspace), xmax = maximum(mspace),
            enlarge_x_limits = 0.01,
            xtick = mtick, xticklabels = mticklabels,
            grid = "both", ymin = 0, ymax = 1,
            enlarge_y_limits = 0.01,
            width = raw"0.5\linewidth", height = raw"0.5\linewidth", leftopts...
        })



        lowerplot = @pgf Plot(
            { line_width = LINE_WIDTH, color = colors[2] }, 
            Coordinates(mspace[1:right], lowerpolicy)
        )

        lowermarker = @pgf Plot(
            { only_marks, color = colors[2], forget_plot }, 
            Coordinates(mspace[[right]], lowerpolicy[[end]])
        )

        push!(policyax, lowerplot, lowermarker)

        if i > 1 
            push!(policyax, LegendEntry(raw"\footnotesize Low $T$"))
        end
        
        upperplot = @pgf Plot(
            { line_width = LINE_WIDTH, color = colors[1] }, 
            Coordinates(mspace[left:end], upperpolicy)
        )

        uppermarker = @pgf Plot(
            { only_marks, forget_plot, color = colors[1] }, 
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
        PGFPlotsX.save(joinpath(plotpath, "policyslice.tikz"), policyfig; include_preamble = true)
    end
    
    policyfig
end

# -- Make simulation of optimal trajectories
begin
    TRAJECTORIES = 10_000;
    simulations = Dict{AbstractModel, EnsembleSolution}();
    X₀ = [Hogg().T₀, log(Hogg().M₀), log(Economy().Y₀)];

    u₀ = [X₀..., 0., 0., 0.] # Introduce three 0s for costs

    for model in tippingmodels
        itp = interpolations[model];

        policies = (itp[:χ], itp[:α]);
        parameters = (model, policies, calibration)

        problem = SDEProblem(Fbreakdown!, Gbreakdown!, u₀, (0., 200.), parameters)

        ensembleprob = EnsembleProblem(problem)

        simulation = solve(ensembleprob; trajectories = TRAJECTORIES);
        println("Done with simulation of $model")

        simulations[model] = simulation
    end
end

begin
    simfig = @pgf GroupPlot({
        group_style = {
            group_size = "$(length(simulations)) by 2",
            horizontal_sep = raw"1em",
        }});

    SMOOTH_FACTOR = 5
    yearticks = 0:20:PLOT_HORIZON

    medianopts = @pgf { line_width = LINE_WIDTH }
    confidenceopts = @pgf { draw = "none", forget_plot }
    fillopts = @pgf { fill = "gray", opacity = 0.5 }
    figopts = @pgf { width = raw"0.5\textwidth", height = raw"0.35\textwidth", grid = "both", xmin = 0, xmax = PLOT_HORIZON }

    qs = [0.01, 0.5, 0.99]

    temperatureticks = makedeviationtickz(0., 6., model; step = 1, digits = 0)

    for (k, model) in enumerate(tippingmodels)
        abatedsol = simulations[model];
        itp = interpolations[model];
        αitp = itp[:α];

        # Abatement expenditure figure
        Efn = @closure (T, m, y, _, _, _, t) -> begin
            abatement = αitp(T, m, t)
            return ε(t, exp(m), abatement, model, calibration)
        end

        EM = computeonsim(abatedsol, Efn, yearlytime);

        Equantiles = timequantiles(EM, qs);
        smoothquantile!.(eachcol(Equantiles), SMOOTH_FACTOR)

        Emedianplot = @pgf Plot(medianopts, Coordinates(yearlytime, Equantiles[:, 2]))
        Elower = @pgf Plot({ confidenceopts..., name_path = "lower" }, Coordinates(yearlytime, Equantiles[:, 1]))
        Eupper = @pgf Plot({ confidenceopts..., name_path = "upper" }, Coordinates(yearlytime, Equantiles[:, 3]))

        Efill = @pgf Plot(fillopts, raw"fill between [of=lower and upper]")

        Eoptionfirst = @pgf k > 1 ? {
            yticklabel = raw"\empty"
        } : {
            ylabel = L"Abated fraction $\varepsilon_t$"
        }

        @pgf push!(simfig, {figopts...,
            title = labelsbymodel[model], xticklabel = raw"\empty", ymin = 0, ymax = 1.03,
            scaled_y_ticks = false, Eoptionfirst...,
        }, Emedianplot, Elower, Eupper, Efill)
    end;

    # Makes the emissions plots in the second row
    for (k, model) in enumerate(tippingmodels)
        abatedsol = simulations[model];
        itp = interpolations[model];
        αitp = itp[:α];

        paths = EnsembleAnalysis.timeseries_point_quantile(abatedsol, qs, yearlytime)
        Tpaths = first.(paths.u)

        Tmedianplot = @pgf Plot(medianopts, Coordinates(yearlytime, getindex.(Tpaths, 2)));
        Tlowerplot = @pgf Plot({ confidenceopts..., name_path = "lower" }, Coordinates(yearlytime, getindex.(Tpaths, 1)));
        Tupperplot = @pgf Plot({ confidenceopts..., name_path = "upper" }, Coordinates(yearlytime, getindex.(Tpaths, 3)));

        Tfill = @pgf Plot(fillopts, raw"fill between [of=lower and upper]")

        figticks = yearticks[1:(k > 1 ? end : end - 1)];

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
            xticklabel_style = { rotate = 45 },
            Toptionfirst...
        }, Tmedianplot, Tlowerplot, Tupperplot, Tfill)
    end;

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "simfig.tikz"), simfig; include_preamble = true)
    end

    simfig
end

begin
    costfig = @pgf GroupPlot({
        group_style = {
            group_size = "$(length(simulations)) by 1",
            horizontal_sep = raw"2em"
    }});

    decadetime = 0:10:PLOT_HORIZON
    decadeslabels = ["$(2020 + dec)s" for  dec in Int.(decadetime[1:end - 1])]

    patterns = [nothing, "crosshatch", "horizontal lines"]
    labels = ["Climate Damages", "Abatement", "Adjustment costs"]
    idxs = [6, 5, 4]

    ytick = 0:0.025:0.1
    yticklabels = [L"%$(y * 100)\%" for y in ytick]

    figopts = @pgf { 
        width = raw"0.5\textwidth", height = raw"0.5\textwidth", grid = "both", 
        symbolic_x_coords = decadeslabels,
        xticklabel_style = { rotate = 45, align = "right" }, xtick = "data",
        enlarge_x_limits = 0.1,
        ymin = 0, ymax = maximum(ytick),
        ybar_stacked, bar_width = "2.5ex",
        x = "3ex", ytick = ytick, yticklabels = yticklabels, reverse_legend,
        scaled_y_ticks = false
    } 

    for (k, model) in enumerate(tippingmodels)
        sim = simulations[model];

        decadespath = EnsembleAnalysis.timeseries_point_median(sim, decadetime)
        decadechange = diff(decadespath.u) / step(decadetime) # average

        kopts = @pgf k > 1 ? {
            yticklabels = raw"\empty"
        } : {
            ylabel = L"\small Average decade costs, \% $Y_t$"
        }

        barchart = @pgf Axis({
            figopts..., kopts...,
            title = labelsbymodel[model]
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
        PGFPlotsX.save(joinpath(plotpath, "opt-costs.tikz"), costfig; include_preamble = true)
    end

    costfig
end