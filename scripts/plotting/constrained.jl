using Revise
using Suppressor: @suppress
using JLD2, UnPack
using FastClosures

using Random

using Plots, PGFPlotsX
using LaTeXStrings, Printf, Colors, ColorSchemes

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")

using Statistics
using Model, Grid
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations: Extrapolation

includet("utils.jl")
includet("../utils/saving.jl")
includet("../utils/simulating.jl")

SAVEFIG = false;
ALLOWNEGATIVE = false;
datapath = "data/simulation-medium";
experimentpath = "data/experiments/simulation-medium";
PLOT_HORIZON = 80.

begin # Import results and interpolations
    simulationfilespath = joinpath(datapath, ALLOWNEGATIVE ? "negative" : "constrained")
    experimentfilespath = joinpath(experimentpath, ALLOWNEGATIVE ? "negative" : "constrained")
    
    simulationfiles = listfiles(simulationfilespath)

    models = AbstractModel[]
    interpolations = Dict{AbstractModel, Dict{Symbol, Extrapolation}}();
    simulations = Dict{AbstractModel, EnsembleSolution}();

    for filepath in simulationfiles
        result = loadtotal(filepath)
        model = last(result)
        push!(models, model)

        interpolations[model] = buildinterpolations(result)

        experimentfilepath = replace(filepath, simulationfilespath => experimentfilespath)

        if isfile(experimentfilepath)
            simulations[model] = JLD2.load_object(experimentfilepath)
        else
            @warn "No simulation found for $model in $experimentfilepath."
        end
    end
end


begin # Extract relevant models
    θ = 10.
    ψ = 0.75
    DamageType = Model.GrowthDamages

    simmodels = filter(
        model -> begin
            model.damages isa DamageType &&
            model.preferences.θ == θ &&
            model.preferences.ψ == ψ
        end, models)

    sort!(tippingmodels; by = model -> model.albedo.Tᶜ)

    jumpmodels = filter(model -> model isa JumpModel, simmodels)

    labels = ["Imminent"] # ["Imminent", "Distant", "Benchmark"]
    labelsbymodel = Dict{AbstractModel, String}(simmodels .=> labels)

    thresholds = unique((model.albedo.Tᶜ for model in tippingmodels))
    PALETTE = colorschemes[:grays]
    graypalette = n -> n > 1 ? get(PALETTE, range(0.1, 0.8; length = n)) : 0.8

    TEMPLABEL = L"Temperature deviations $T_t - T^{p}$"
    LINE_WIDTH = 2.5
    defopts = @pgf { line_width = LINE_WIDTH }

    ΔTmax = 5.
    ΔTspace = range(0., ΔTmax; length = 51)
    Tspace = ΔTspace .+ Hogg().Tᵖ
    Tmin, Tmax = extrema(Tspace)

    calibration = JLD2.load_object("data/calibration.jld2")

    yearlytime = range(0., PLOT_HORIZON; step = 1 / 3) |> collect

    temperatureticks = makedeviationtickz(0., ΔTmax, first(models); step = 1, digits = 0)
end;

begin
    simfig = @pgf GroupPlot({
        group_style = { 
            group_size = "$(length(simmodels)) by 2",
            horizontal_sep = raw"1em",
            vertical_sep = raw"2em"
        }
    });

    yearticks = 0:20:PLOT_HORIZON

    βextrema = (0., 0.15)
    βticks = range(βextrema...; step = 0.05) 
    βticklabels = [@sprintf("%.0f \\%%", 100 * y) for y in βticks]

    emissionsextrema = (0., Model.Eᵇ(0, calibration))
    emissionsticks = range(emissionsextrema...; step = 10)
    emissionsticklabels = emissionsticks

    confidenceopts = @pgf { opacity = 0.5 }
    figopts = @pgf { width = raw"0.33\textwidth", height = raw"0.3\textwidth", grid = "both", xmin = 0, xmax = PLOT_HORIZON }

    qs = [0.1, 0.5, 0.9]

    # Makes the β plots in the first row
    for (k, model) in enumerate(simmodels)
        abatedsol = simulations[model];
        itp = interpolations[model];
        αitp = itp[:α];

        # Abatement expenditure figure
        βfn = @closure (T, m, y, t) -> begin
            abatement = αitp(T, m, t)
            emissivity = ε(t, exp(m), abatement, model)
            return β(t, emissivity, model.economy)
        end

        βM = computeonsim(abatedsol, βfn, yearlytime);
       
        βquantiles = timequantiles(βM, qs);
        smoothquantile!.(eachcol(βquantiles), 30)

        βmedianplot = @pgf Plot(defopts, Coordinates(yearlytime, βquantiles[:, 2]))
        βlowerplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, βquantiles[:, 1]))
        βupperplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, βquantiles[:, 3]))

        βoptionfirst = @pgf k > 1 ? {
            yticklabel = raw"\empty"
        } : { 
            ylabel = L"Abatement as \% of $Y_t$",
            ytick = βticks, yticklabels = βticklabels
        }

        @pgf push!(simfig, {figopts...,
            ymin = βextrema[1], ymax = βextrema[2],
            title = labelsbymodel[model], xticklabel = raw"\empty",
            scaled_y_ticks = false, βoptionfirst...,
        }, βmedianplot, βlowerplot, βupperplot)
    end;

    # Makes the emissions plots in the second row
    for (k, model) in enumerate(simmodels)
        abatedsol = simulations[model];
        itp = interpolations[model];
        αitp = itp[:α];

        # Abatement expenditure figure
        emissions = @closure (T, m, y, t) -> begin
            abatement = αitp(T, m, t)
            emissivity = ε(t, exp(m), abatement, model)
            return Model.Eᵇ(t, model.calibration) * (1 - emissivity)
        end

        emissionsM = computeonsim(abatedsol, emissions, yearlytime);
       
        emissionsquantiles = timequantiles(emissionsM, qs);
        smoothquantile!.(eachcol(emissionsquantiles), 30)

        emissionsmedianplot = @pgf Plot(defopts, Coordinates(yearlytime, emissionsquantiles[:, 2]))
        emissionslowerplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, emissionsquantiles[:, 1]))
        emissionsupperplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, emissionsquantiles[:, 3]))

        emissionsoptionfirst = @pgf k > 1 ? {
            yticklabel = raw"\empty"
        } : { 
            ylabel = "Emissions, Gt",
            ytick = emissionsticks, yticklabels = emissionsticklabels
        }

        @pgf push!(simfig, {figopts...,
            ymin = emissionsextrema[1], ymax = emissionsextrema[2],
            xticklabel = raw"\empty",
            scaled_y_ticks = false,
            xtick = yearticks,
            xticklabels = 2020 .+ yearticks,
            xticklabel_style = { rotate = 45 }, 
            emissionsoptionfirst...,
        }, emissionsmedianplot, emissionslowerplot, emissionsupperplot)
    end;


    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "simfig.tikz"), simfig; include_preamble = true)
    end

    simfig
end

if false
    # --- Plotting the regret problem. Discover tipping point only after T ≥ Tᶜ.
    begin    
        Tmin, Tmax = (1., 5)
        Tregretticks = makedeviationtickz(Tmin, Tmax, first(models); step = 1, digits = 0)

        regretfig = @pgf GroupPlot({
            group_style = { 
                group_size = "1 by 2",
                horizontal_sep = raw"1em",
                vertical_sep = raw"2em"
            }
        });

        regfigopts = @pgf { width = raw"0.42\textwidth", height = raw"0.3\textwidth", grid = "both", xmin = 0., xmax = maximum(yearlytime) }

        modelimminent, modelremote = tippingmodels

        βregret = @closure (T, m, y, t) -> begin
            model = ifelse(T - modelimminent.hogg.Tᵖ ≥ modelimminent.albedo.Tᶜ, modelimminent, modelremote)
            αitp = interpolations[model][:α]

            return β(t, ε(t, exp(m), αitp(T, m, t), model), model.economy)
        end

        βregextrema = (0., 0.08)
        βregticks = range(βregextrema...; step = 0.02) |> collect
        βregtickslabels = [@sprintf("%.0f \\%%", 100 * y) for y in βregticks]

        # Abatement expenditure figure
        βM = computeonsim(regretsolution, βregret, yearlytime)
        βquantiles = timequantiles(βM, qs)
        smoothquantile!.(eachcol(βquantiles), 30)

        βmedianplot = @pgf Plot(defopts, Coordinates(yearlytime, βquantiles[:, 2]))
        βlowerplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, βquantiles[:, 1]))
        βupperplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, βquantiles[:, 3]))


        @pgf push!(regretfig, {regfigopts...,
            ymin = βregextrema[1], ymax = βregextrema[2],
            ytick = βregticks,
            scaled_y_ticks = false,
            yticklabels = βregtickslabels,
            ylabel = L"Abatement, \% of $Y_t$",
            xtick = yearticks,
            xticklabels = raw"\empty"
        }, βmedianplot, βlowerplot, βupperplot)

        # Temperature figure
        paths = EnsembleAnalysis.timeseries_point_quantile(regretsolution, qs, yearlytime)
        Tpaths = first.(paths.u)

        Tmedianplot = @pgf Plot(defopts, Coordinates(yearlytime, getindex.(Tpaths, 2)))
        Tlowerplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, getindex.(Tpaths, 1)))
        Tupperplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, getindex.(Tpaths, 3)))

        @pgf push!(regretfig, {regfigopts...,
            ymin = Tmin + modelimminent.hogg.Tᵖ, 
            ymax = Tmax + modelimminent.hogg.Tᵖ,
            ytick = Tregretticks[1], yticklabels = Tregretticks[2],
            ylabel = raw"Temperature $T_t$",
            xtick = yearticks,
            xticklabels = 2020 .+ yearticks,
            xticklabel_style = { rotate = 45 }
        }, Tmedianplot, Tlowerplot, Tupperplot)

        if SAVEFIG
            PGFPlotsX.save(joinpath(plotpath, "regretfig.tikz"), regretfig; include_preamble = true)
        end

        regretfig
    end

    # --- Plot of highest path in regret
    yearlyregrets = regretsolution(yearlytime);
    _, maxidx = findmax(v -> maximum(first.(v.u)), yearlyregrets);
    maxsim = yearlyregrets[maxidx];

    _, minidx = findmin(v -> maximum(first.(v.u)), yearlyregrets);
    minsim = yearlyregrets[minidx];

    begin # Computes the nullclines
        nullclinevariation = Dict{Float64,Vector{Vector{NTuple{2,Float64}}}}()
        for model in reverse(tippingmodels)
            nullclines = Vector{NTuple{2,Float64}}[]

            currentM = NTuple{2,Float64}[]
            currentlystable = true

            for T in Tspace
                M = Model.Mstable(T, model.hogg, model.albedo)
                isstable = Model.radiativeforcing′(T, model.hogg, model.albedo) < 0
                if isstable == currentlystable
                    push!(currentM, (M, T))
                else
                    currentlystable = !currentlystable
                    push!(nullclines, currentM)
                    currentM = [(M, T)]
                end
            end

            push!(nullclines, currentM)
            nullclinevariation[model.albedo.Tᶜ] = nullclines
        end
    end

    begin # Plot of breaking regret
        Mmax = 600.

        breakfig = @pgf Axis({
            width = raw"0.9\textwidth",
            height = raw"0.7\textwidth",
            grid = "both",
            ylabel = TEMPLABEL,
            xlabel = raw"Carbon concentration $M_t$",
            xmin = tippingmodels[1].hogg.Mᵖ, xmax = Mmax,
            xtick = 200:100:Mmax,
            yticklabels = temperatureticks[2],
            ytick = temperatureticks[1],
            ymin = Tmin + hogg.Tᵖ, ymax = Tmax + hogg.Tᵖ,
            legend_cell_align = "left"
        })

        for model in reverse(tippingmodels) # Nullcline plots
            Tᶜ = model.albedo.Tᶜ
            color = thresholdscolors[Tᶜ]

            stableleft, unstable, stableright = nullclinevariation[Tᶜ]

            leftcurve = @pgf Plot({color = color, line_width = LINE_WIDTH}, Coordinates(stableleft))
            unstablecurve = @pgf Plot({color = color, line_width = LINE_WIDTH, forget_plot, dotted}, Coordinates(unstable))
            rightcurve = @pgf Plot({color = color, line_width = LINE_WIDTH, forget_plot}, Coordinates(stableright))

            label = labelsbythreshold[Tᶜ]
            legend = LegendEntry(label)

            push!(breakfig, leftcurve, legend, unstablecurve, rightcurve)
        end

        colors = ["red", "blue"]

        for (k, sim) in enumerate([maxsim, minsim])

            simcoords = Coordinates(exp.(getindex.(sim.u, 2)), getindex.(sim.u, 1))
            curve = @pgf Plot({line_width = LINE_WIDTH / 2, opacity = 0.7, color = colors[k]}, simcoords)
            markers = @pgf Plot({only_marks, mark_options = {fill = "black", scale = 1.5, draw_opacity = 0, opacity = 0.7, color = colors[k]}, mark_repeat = 20}, simcoords)

            push!(breakfig, curve, markers)
        end

        breakfig
    end
end