using Revise
using Distributed
using JLD2, DotEnv, CSV
using UnPack
using DataFrames, DataStructures

using Plots, Printf, PGFPlotsX, Colors, ColorSchemes
using Statistics, LaTeXStrings

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")

@everywhere begin
    using FastClosures

    using DifferentialEquations, DifferentialEquations.EnsembleAnalysis
    using Interpolations

    using Model, Grid

    include("../utils/saving.jl")
    include("../utils/simulating.jl")
    include("utils.jl")
end

println("Simulating with $(nprocs()) processes...")

begin # Global variables
    env = DotEnv.config(".env")
    BASELINE_YEAR = 2020

    DATAPATH = get(env, "DATAPATH", "data")
    
    datapath = joinpath(DATAPATH, get(env, "SIMULATIONPATH", "simulaton"), "")

    PLOTPATH = get(env, "PLOTPATH", "plots")
    PRESENTATIONPATH = joinpath(PLOTPATH, "presentation")

    SAVEFIG = false
    LINE_WIDTH = 2.5
    SEED = 11148705
end;

begin # Construct models and grids
    thresholds = [1.5, 2.5];

	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	preferences = EpsteinZin()
    damages = GrowthDamages()
    economy = Economy()
    jump = Jump()
    hogg = Hogg()
    
	models = AbstractModel[]

	for Tᶜ ∈ thresholds
	    albedo = Albedo(Tᶜ)
	    model = TippingModel(albedo, hogg, preferences, damages, economy, calibration)

		push!(models, model)

	end

    jumpmodel = JumpModel(jump,  hogg, preferences, damages, economy, calibration)
    push!(models, jumpmodel)
end;

begin # Labels, colors and axis
    PALETTE = colorschemes[:grays]
    graypalette = n -> n > 1 ? get(PALETTE, range(0.1, 0.8; length = n)) : 0.8

    thresholdcolor = Dict(thresholds .=> graypalette(length(thresholds)))

    rawlabels = [ L"\textit{imminent}", L"\textit{remote}", L"\textit{benchmark}"]

    labels = Dict{AbstractModel, LaTeXString}(models .=> rawlabels[eachindex(models)])

    TEMPLABEL = L"Temperature deviations $T_t - T^{p}$"
    defopts = @pgf { line_width = LINE_WIDTH }

    ΔTmax = 8.
    ΔTspace = range(0., ΔTmax; length = 51)
    Tspace = ΔTspace .+ Hogg().Tᵖ
    Tmin, Tmax = extrema(Tspace)

    horizon = round(Int64, last(calibration.tspan))
    yearlytime = range(0., horizon; step = 1 / 3) |> collect

    temperatureticks = makedeviationtickz(0., ΔTmax, first(models); step = 1, digits = 0)

    X₀ = [hogg.T₀, log(hogg.M₀)]

    baufn = SDEFunction(Fbau!, G!)
end;

# --- Optimal emissions 
begin
    results = loadtotal.(models; datapath);
    itps = buildinterpolations.(results);
    modelmap = Dict{AbstractModel, typeof(first(itps))}(models .=> itps);
    abatementmap = Dict{AbstractModel, Interpolations.Extrapolation}(model => itp[:α] for (model, itp) in modelmap)
end;

begin # Solve an ensemble problem for all models with the bau scenario
    trajectories = 10_000
    ratejump = VariableRateJump(rate, tippingopt!);
    sols = Dict{AbstractModel, EnsembleSolution}()

    for model in models
        itp = modelmap[model];
        αitp = itp[:α];

        prob = SDEProblem(F!, G!, X₀, (0., horizon), (model, αitp))

        if isa(model, JumpModel)
            jumpprob = JumpProblem(prob, ratejump)
            ensprob = EnsembleProblem(jumpprob)
            abatedsol = solve(ensprob, SRIW1(), EnsembleDistributed(); trajectories)
        else
            ensprob = EnsembleProblem(prob)
            abatedsol = solve(ensprob, EnsembleDistributed(); trajectories)
        end

        sols[model] = abatedsol
    end
end;

# Constructs a Group plot, one for the path of T and one for β
begin
    simfig = @pgf GroupPlot({
        group_style = { 
            group_size = "2 by $(length(models))",
            horizontal_sep = raw"0.15\textwidth",
            vertical_sep = raw"0.1\textwidth"
        }, width = raw"\textwidth", height = raw"\textwidth",
    });

    yearticks = 0:20:horizon

    βextrema = (0., 0.03)
    βticks = range(βextrema...; step = 0.01) |> collect
    βticklabels = [@sprintf("%.0f \\%%", 100 * y) for y in βticks]

    Textrema = (1., 2.)
    Tticks = makedeviationtickz(Textrema..., first(models); step = 0.5, digits = 1)

    confidenceopts = @pgf { dotted, opacity = 0.5 }
    figopts = @pgf { width = raw"0.45\textwidth", height = raw"0.3\textwidth", grid = "both", xmin = 0, xmax = horizon }

    for (k, model) in enumerate(models)
        abatedsol = sols[model]
        itp = modelmap[model]
        αitp = itp[:α]

        # Abatement expenditure figure
        βM = computeonsim(abatedsol, 
        (T, m, t) -> β(t, ε(t, exp(m), αitp(T, m, t), model), model.economy), yearlytime)
       
        βquantiles = timequantiles(βM, [0.1, 0.5, 0.9])
        smoothquantile!.(eachcol(βquantiles), 30)

       βmedianplot = @pgf Plot(defopts, Coordinates(yearlytime, βquantiles[:, 2]))
       βlowerplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, βquantiles[:, 1]))
       βupperplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, βquantiles[:, 3]))


       βoptionfirst = @pgf k > 1 ? {} : { title = L"Abatement as \% of $Y_t$" }
       βoptionlast = @pgf k < length(models) ?
           { 
               xticklabels = raw"\empty"
           } : {
               xtick = yearticks,
               xticklabels = BASELINE_YEAR .+ yearticks,
               xticklabel_style = { rotate = 45 }
           }

       @pgf push!(simfig, {figopts...,
           ymin = βextrema[1], ymax = βextrema[2],
           ytick = βticks,
           scaled_y_ticks = false,
           yticklabels = βticklabels,
           ylabel = labels[model],
           βoptionlast..., βoptionfirst...
       }, βmedianplot, βlowerplot, βupperplot)

        # Temperature figure
        paths = EnsembleAnalysis.timeseries_point_quantile(abatedsol, [0.05, 0.5, 0.95], yearlytime)
        Tpaths = first.(paths.u)

        Tmedianplot = @pgf Plot(defopts, Coordinates(yearlytime, getindex.(Tpaths, 2)))
        Tlowerplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, getindex.(Tpaths, 1)))
        Tupperplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, getindex.(Tpaths, 3)))

        Ttickoptions = @pgf k < length(models) ?
            { 
                xticklabels = raw"\empty"
            } : {
                xtick = yearticks,
                xticklabels = BASELINE_YEAR .+ yearticks,
                xticklabel_style = { rotate = 45 }
            }

        Ttitleoptions = @pgf k > 1 ? {} : { title = raw"Temperature $T_t$" }

        @pgf push!(simfig, {figopts...,
            ymin = Textrema[1] + hogg.Tᵖ, ymax = Textrema[2] + hogg.Tᵖ,
            ytick = Tticks[1], yticklabels = Tticks[2],
            Ttickoptions..., Ttitleoptions...
        }, Tmedianplot, Tlowerplot, Tupperplot)
    end;

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "simfig.tikz"), simfig; include_preamble = true) 
    end

    simfig
end

begin # Solve the regret problem. Discover tipping point only after T ≥ Tᶜ.
    modelimminent, modelremote = models[[1, 2]]
    αimminent = abatementmap[modelimminent]
    αremote = abatementmap[modelremote]

    parameters = ((modelimminent, αimminent), (modelremote, αremote));

    regretprob = SDEProblem(Fregret!, G!, X₀, (0., horizon), parameters) |> EnsembleProblem

    regretsol = solve(regretprob, EnsembleDistributed(); trajectories)
end;

begin
    regretfig = @pgf GroupPlot({
        group_style = { 
            group_size = "2 by 1",
            horizontal_sep = raw"0.15\textwidth",
            vertical_sep = raw"0.1\textwidth"
        }, width = raw"\textwidth", height = raw"\textwidth",
    });


    βregret = @closure (T, m, t) -> begin
        model = ifelse(T - hogg.Tᵖ > modelimminent.albedo.Tᶜ, modelimminent, modelremote)

        αitp = abatementmap[model]

        return β(t, ε(t, exp(m), αitp(T, m, t), model), model.economy)
    end

    βregextrema = (0., 0.08)
    βregticks = range(βregextrema...; step = 0.02) |> collect
    βregtickslabels = [@sprintf("%.0f \\%%", 100 * y) for y in βregticks]

    # Abatement expenditure figure
    βM = computeonsim(regretsol, βregret, yearlytime)
    
    βquantiles = timequantiles(βM, [0.1, 0.5, 0.9])
    smoothquantile!.(eachcol(βquantiles), 30)

    βmedianplot = @pgf Plot(defopts, Coordinates(yearlytime, βquantiles[:, 2]))
    βlowerplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, βquantiles[:, 1]))
    βupperplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, βquantiles[:, 3]))


    @pgf push!(regretfig, {figopts...,
        ymin = βregextrema[1], ymax = βregextrema[2],
        ytick = βregticks,
        scaled_y_ticks = false,
        yticklabels = βregtickslabels,
        title = L"Abatement as \% of $Y_t$",
        xtick = yearticks,
        xticklabels = BASELINE_YEAR .+ yearticks,
        xticklabel_style = { rotate = 45 }
    }, βmedianplot, βlowerplot, βupperplot)

    # Temperature figure
    paths = EnsembleAnalysis.timeseries_point_quantile(regretsol, [0.01, 0.5, 0.99], yearlytime)
    Tpaths = first.(paths.u)

    Tmedianplot = @pgf Plot(defopts, Coordinates(yearlytime, getindex.(Tpaths, 2)))
    Tlowerplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, getindex.(Tpaths, 1)))
    Tupperplot = @pgf Plot({ defopts..., confidenceopts... }, Coordinates(yearlytime, getindex.(Tpaths, 3)))

    @pgf push!(regretfig, {figopts...,
        ymin = Textrema[1] + hogg.Tᵖ, ymax = Textrema[2] + hogg.Tᵖ,
        ytick = Tticks[1], yticklabels = Tticks[2],
        title = raw"Temperature $T_t$",
        xtick = yearticks,
        xticklabels = BASELINE_YEAR .+ yearticks,
        xticklabel_style = { rotate = 45 }
    }, Tmedianplot, Tlowerplot, Tupperplot)

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "regretfig.tikz"), regretfig; include_preamble = true)
    end

    regretfig
end