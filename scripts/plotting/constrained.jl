using Revise
using JLD2, DotEnv, CSV, UnPack
using FastClosures

using Plots
using PGFPlotsX
using LaTeXStrings, Printf, Colors, ColorSchemes

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")

using Statistics
using Model, Grid
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations: Extrapolation

include("utils.jl")
include("../utils/saving.jl")
include("../utils/simulating.jl")

begin # Environment variables
    env = DotEnv.config()
    datapath = get(env, "DATAPATH", "data")
    simulationpath = get(env, "SIMULATIONPATH", "simulations")

    plotpath = get(env, "PLOTPATH", "plots")

    BASELINE_YEAR = 2020
    SAVEFIG = false

    calibration = load_object(joinpath(datapath, "calibration.jld2"))

    experimentpath = joinpath(datapath, "experiments", "constrained.jld2")
    @assert ispath(experimentpath)

    regretpath = joinpath(datapath, "experiments", "regret-constrained.jld2")
    @assert ispath(regretpath)
end;

begin # Load simulations and interpolations 
    @load experimentpath solutions;
    allmodels = collect(keys(solutions));
    tippingmodels = filter(model -> model isa TippingModel, allmodels);
    sort!(tippingmodels; by = model -> model.albedo.Tᶜ);

    jumpmodels = filter(model -> model isa JumpModel, allmodels);
    models = AbstractModel[tippingmodels..., jumpmodels...];

    results = loadtotal.(models; datapath = joinpath(datapath, simulationpath))
    
    interpolations = buildinterpolations.(results)
    itpsmap = Dict{AbstractModel, Dict{Symbol, Extrapolation}}(models .=> interpolations)
end;

begin # Labels, colors and axis
    thresholds = sort([model.albedo.Tᶜ for model in tippingmodels])

    PALETTE = colorschemes[:grays]
    graypalette = n -> n > 1 ? get(PALETTE, range(0.1, 0.8; length = n)) : 0.8

    thresholdscolors = Dict(thresholds .=> graypalette(length(thresholds)))

    rawlabels = [ "Imminent", "Remote", "Benchmark"]
    labels = Dict{AbstractModel, String}(models .=> rawlabels)

    TEMPLABEL = L"Temperature deviations $T_t - T^{p}$"
    defopts = @pgf { line_width = 2.5 }

    ΔTmax = 8.
    ΔTspace = range(0., ΔTmax; length = 51)
    Tspace = ΔTspace .+ Hogg().Tᵖ
    Tmin, Tmax = extrema(Tspace)

    horizon = round(Int64, last(calibration.tspan))
    yearlytime = range(0., horizon; step = 1 / 3) |> collect

    temperatureticks = makedeviationtickz(0., ΔTmax, first(models); step = 1, digits = 0)
end;

# Constructs a Group plot, one for the path of T and β
begin
    simfig = @pgf GroupPlot({
        group_style = { 
            group_size = "$(length(models)) by 3",
            horizontal_sep = raw"1em",
            vertical_sep = raw"2em"
        }
    });

    yearticks = 0:20:horizon

    βextrema = (0., 0.03)
    βticks = range(βextrema...; step = 0.01) |> collect
    βticklabels = [@sprintf("%.0f \\%%", 100 * y) for y in βticks]

    Textrema = (1., 2.5)
    Tticks = makedeviationtickz(Textrema..., first(models); step = 1, digits = 0)

    confidenceopts = @pgf { opacity = 0.5 }
    figopts = @pgf { width = raw"0.33\textwidth", height = raw"0.3\textwidth", grid = "both", xmin = 0, xmax = horizon }

    qs = [0.1, 0.5, 0.9]

    # Makes the β plots in the first row
    for (k, model) in enumerate(models)
        abatedsol = solutions[model]
        itp = itpsmap[model]
        αitp = itp[:α]

        # Abatement expenditure figure
        βfn = @closure (T, m, y, t) -> begin
            abatement = αitp(T, m, t)
            emissivity = ε(t, exp(m), abatement, model)
            return β(t, emissivity, model.economy)
        end

        βM = computeonsim(abatedsol, βfn, yearlytime)
       
        βquantiles = timequantiles(βM, qs)
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
            title = labels[model], xticklabel = raw"\empty",
            scaled_y_ticks = false, βoptionfirst...,
        }, βmedianplot, βlowerplot, βupperplot)
    end;

    # Makes the T plots in the second row
    for (k, model) in enumerate(models)
        abatedsol = solutions[model]
        paths = EnsembleAnalysis.timeseries_point_quantile(abatedsol, qs, yearlytime)
        Tpaths = first.(paths.u)

        Tmedianplot = @pgf Plot(defopts, Coordinates(yearlytime, getindex.(Tpaths, 2)))
        Tlowerplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, getindex.(Tpaths, 1)))
        Tupperplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, getindex.(Tpaths, 3)))

        Ttitleoptions = @pgf k > 1 ? {
            yticklabel = raw"\empty"
        } : { ylabel = raw"Temperature $T_t$" }

        @pgf push!(simfig, {figopts...,
            ymin = Textrema[1] + model.hogg.Tᵖ, 
            ymax = Textrema[2] + model.hogg.Tᵖ,
            ytick = Tticks[1], yticklabels = Tticks[2],
            xtick = yearticks,
            xticklabels = BASELINE_YEAR .+ yearticks,
            xticklabel_style = { rotate = 45 },
            Ttitleoptions...
        }, Tmedianplot, Tlowerplot, Tupperplot)
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "simfig.tikz"), simfig; include_preamble = true)
    end

    simfig
end

# --- Plotting the regret problem. Discover tipping point only after T ≥ Tᶜ.
@load regretpath regretsolution;

begin
    
    Tmin, Tmax = (1., 2.5)
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
        αitp = itpsmap[model][:α]

        return β(t, ε(t, exp(m), αitp(T, m, t), model), model.economy)
    end

    βregextrema = (0., 0.06)
    βregticks = range(βregextrema...; step = 0.01) |> collect
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
        ylabel = L"Abatement as \% of $Y_t$",
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
        xticklabels = BASELINE_YEAR .+ yearticks,
        xticklabel_style = { rotate = 45 }
    }, Tmedianplot, Tlowerplot, Tupperplot)

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "regretfig.tikz"), regretfig; include_preamble = true)
    end

    regretfig
end