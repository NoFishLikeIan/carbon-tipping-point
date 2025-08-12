using Revise
using JLD2, CSV, DotEnv
using UnPack
using DataFrames, DataStructures
using FastClosures
using StatsBase

using DifferentialEquations, DifferentialEquations.EnsembleAnalysis

using Plots, Printf, PGFPlotsX, Colors, ColorSchemes, LaTeXStrings
using Statistics

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{siunitx}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\DeclareSIUnit{\ppm}{p.p.m.}")

using Model, Grid

includet("../../utils/saving.jl")
includet("../../utils/simulating.jl")
includet("../utils.jl")

begin # Global variables
    BASELINE_YEAR = 2020
    PLOT_HORIZON = 80.
    DATAPATH = "data"
    PLOTPATH = "plots/game"
    PRESENTATIONPATH = joinpath(PLOTPATH, "presentation")
    THESISPATH = "../phd-thesis/climate-plots/game"

    SAVEFIG = false
    LINE_WIDTH = 2.5
    SEED = 11148705

    LABELS = ["OECD", "RoW"]
    TJOINTLABEL = L"Temperature $T_{i, t} \; [\si{\degree}]$"
    TLABELS = [L"Temperature $T_{%$i, t} \; [\si{\degree}]$" for i in LABELS]
    MLABEL = L"Carbon concentration $M_t \; [\si{\ppm}]$"
end;


begin # Environment variables
    env = DotEnv.config(".env.game")
    plotpath = get(env, "PLOTPATH", "plots")
    datapath = get(env, "DATAPATH", "data")
    simulationpath = get(env, "SIMULATIONPATH", "simulations")

    calibration = load_object(joinpath(datapath, "calibration.jld2"))
    regionalcalibration = load_object(joinpath(datapath, "regionalcalibration.jld2"))
end;

begin # Models definition
    # -- Climate
    hogg = Hogg()
    # -- Economy and Preferences
    preferences = EpsteinZin();
    oecdeconomy, roweconomy = RegionalEconomies()
    damages = Kalkuhl()

    oecdmodel = LinearModel(hogg, preferences, damages, oecdeconomy)

    rowmodels = SortedDict{Float64, AbstractModel}()    
    thresholds = [1.8, 2., 2.5]

    for threshold in thresholds
        rowmodels[threshold] = TippingModel(Albedo(threshold), hogg, preferences, damages, roweconomy);
    end

    rowmodels[Inf] = LinearModel(hogg, preferences, damages, roweconomy)
    push!(thresholds, Inf)
end;

begin # Labels, colors and axis
    PALETTE = cgrad(:Reds, rev = true)
    colors = get(PALETTE, range(0, 0.5; length = length(rowmodels)))
    
    oecdcolor = RGB(2 / 255, 57 / 255, 74 / 255)

    colorsbymodels = Dict{AbstractModel, RGB{Float64}}(values(rowmodels) .=> colors)

    ΔTmax = 6.
    ΔTspace = range(0.0, ΔTmax; length = 51)
    Tspace = ΔTspace .+ Hogg().Tᵖ

    horizon = round(Int64, calibration.tspan[2])
    yearlytime = 0:1:horizon

    temperatureticks = collect.(makedeviationtickz(0.0, ΔTmax, oecdmodel; step=1, digits=0))

    Tmin, Tmax = extrema(temperatureticks[1])
end;

begin # Albedo plot
    ytick = 0.28:0.02:0.32

    albedofig = @pgf Axis(
        {
        width = raw"0.5\textwidth",
        height = raw"0.5\textwidth",
        grid = "both",
        xlabel = TLABELS[2],
        ylabel = "Positive feedback \$\\lambda(T_t)\$",
        xticklabels = temperatureticks[2],
        xtick = temperatureticks[1],
        xmin = Tmin, xmax = Tmax,
        ymin = ytick[1] - 0.01, ymax = ytick[end] + 0.01,
        ytick = ytick,
        legend_cell_align = "left"
    })

    if SAVEFIG
        for path in (PLOTPATH, THESISPATH)
            PGFPlotsX.save(joinpath(path, "skeleton-albedo.tikz"), albedofig; include_preamble = true)
        end
    end

    @pgf for (threshold, model) in rowmodels
        if !(model isa TippingModel)
            continue
        end

        loss = [Model.λ(T, model.hogg, model.albedo) for T in Tspace]

        Tᶜ = model.albedo.Tᶜ
        curve = Plot({
                color = colorsbymodels[model], 
                line_width = LINE_WIDTH, opacity = 0.8
            }, Coordinates(zip(Tspace, loss)))

        label = "$threshold °C"
        legend = LegendEntry(label)

        push!(albedofig, curve, legend)
    end

    @pgf albedofig["legend style"] = raw"at = {(0.95, 0.95)}"

    if SAVEFIG
        for path in (PLOTPATH, THESISPATH)
            PGFPlotsX.save(joinpath(path, "albedo.tikz"), albedofig; include_preamble=true)
        end
    end

    albedofig
end

begin
    sims = Dict{AbstractModel, DiffEqArray}()
    quantiles = [0.1, 0.5, 0.9]
    mediandx = findfirst(q -> q == 0.5, quantiles)

    for (threshold, rowmodel) in rowmodels
        models = (oecdmodel, rowmodel)
        parameters = (models, calibration)

        X₀ = [oecdmodel.hogg.T₀, rowmodel.hogg.T₀, log(oecdmodel.hogg.M₀)]

        prob = SDEProblem(Fbau!, G!, X₀, (0.0, 200.0), parameters)
        sol = solve(EnsembleProblem(prob), trajectories = 10_000)

        println("Done with simulation of threshold = $threshold")

        simpath = timeseries_point_quantile(sol, quantiles, yearlytime)

        sims[rowmodel] = simpath
    end
end;

begin
    _, samplesim = first(sims)

    Msample = getindex.(samplesim.u, 3)
    Mmedianpath = exp.(getindex.(Msample, mediandx))
    Mticks = Mmedianpath[1:20:end]
    Mmax = ceil(maximum(Mticks), digits = -1)

    Mtickslabels = [
        L"\small $%$M$\\ \footnotesize ($%$y$)"
        for (M, y) in zip(round.(Int, Mticks), 2020:10:2100)
    ]

    baufig = @pgf GroupPlot({
        group_style = {
            group_size = "2 by 2",
            xticklabels_at = "edge bottom",
            yticklabels_at = "edge left",
            vertical_sep = "30pt", horizontal_sep = "10pt"
        }
    })

    for (pdx, entry) in enumerate(rowmodels)
        threshold, rowmodel = entry

        colors = [oecdcolor, colorsbymodels[rowmodel]]
        simpath = sims[rowmodel]

        yopts = @pgf isodd(pdx) ? {
            ylabel = TJOINTLABEL,
            ytick = temperatureticks[1],
            yticklabels = temperatureticks[2]
        } : {
            ytick = temperatureticks[1],
            yticklabels = []
        }

        thirdopts = @pgf pdx == 3 ? {
            xtick = Mticks[1:(end - 1)],
            xticklabels = Mtickslabels[1:(end - 1)],
            xticklabel_style = {align = "center"}
        } : {}

        lastops = @pgf pdx == 4 ? {
            xlabel = MLABEL,
            xtick = Mticks,
            xticklabels = Mtickslabels,
            xticklabel_style = {align = "center"},
            xlabel_style = {xshift = raw"-0.25\textwidth", align = "center"}
        } : {}

        title = isfinite(threshold) ? @sprintf("\$T^c = %.1f \\si{\\degree} \$", threshold) : "No Tipping"

        @pgf push!(baufig, {
            width = raw"0.5\textwidth",
            height = raw"0.5\textwidth",
            grid = "both",
            ymin = Tmin, ymax = Tmax,
            xmin = 380, xmax = 845,
            title = title,
            yopts..., thirdopts..., lastops...
        })

        for jdx in 1:2
            color = colors[jdx]
            Tpath = @. getindex(simpath.u, jdx)
            median = getindex.(Tpath, mediandx)
            mediancoords = Coordinates(Mmedianpath, median)

            medianpath = @pgf Plot({color = color, line_width = LINE_WIDTH}, mediancoords)
            push!(baufig, medianpath, LegendEntry(LABELS[jdx]))

            markers = @pgf Plot({only_marks, mark_options = {fill = "black", scale = 1.5, draw_opacity = 0, color = color}, forget_plot, mark_repeat = 10}, mediancoords)

            push!(baufig, markers)

            for s in 1:(mediandx - 1)
                lowerpath = @pgf Plot(
                    {draw = "none", name_path = "lower_$s", forget_plot}, 
                    Coordinates(Mmedianpath, getindex.(Tpath, mediandx - s)))

                upperpath = @pgf Plot(
                    {draw = "none", name_path = "upper_$s", forget_plot}, 
                    Coordinates(Mmedianpath, getindex.(Tpath, mediandx + s)))

                opacity = 0.15 + 0.2 * (s / (mediandx - 1))

                shading = @pgf Plot({fill = color, opacity = opacity, forget_plot}, "fill between [of=lower_$s and upper_$s]")

                push!(baufig, lowerpath, upperpath, shading)
            end

        end
    end

    @pgf baufig["legend style"] = raw"at = {(0.95, 0.3)}"

    if SAVEFIG
        for path in (PLOTPATH, THESISPATH)
            PGFPlotsX.save(joinpath(path, "baufig.tikz"), baufig; include_preamble=true)
        end
    end

    baufig
end

begin # Simulate carbon concentrations
    mbau(m, model, t) = γ(t, calibration)
    σₘbau(m, model, t) = model.hogg.σₘ

    mfn = SDEFunction(mbau, σₘbau)

    mbauprob = SDEProblem(mfn, log(hogg.M₀), (0.0, 80.0), oecdmodel)
    mensemble = EnsembleProblem(mbauprob)
    mbausims = solve(mensemble; trajectories = 10_000)
end

begin # Growth of carbon concentration 
    horizon = Int(last(yearlytime))
    decadetime = 0:10:(horizon - 1)
    decadeslabels = ["$(2020 + dec)s" for  dec in decadetime]

    ytick = 0:0.05:0.2
    yticklabels = [L"%$(y * 100)\%" for y in ytick]

    γfig = @pgf Axis({
        width = raw"0.8\textwidth", height = raw"0.56\textwidth",
        grid = "both", 
        symbolic_x_coords = decadeslabels,
        xticklabel_style = { rotate = 45, align = "right" }, xtick = "data",
        enlarge_x_limits = 0.1,
        ybar_stacked, bar_width = "6ex",
        x = "7ex", reverse_legend,
        ymin = 0, ymax = maximum(ytick), 
        ytick = ytick, yticklabels = yticklabels,
        scaled_y_ticks = false, ylabel = L"Cumulative decade growth $\int \gamma^b_t \; \text{d}t$ of $M^b_t$"
    })

    decadebegin = (0:10:(horizon - 10)) .+ 1
    growthdata = [γ(t, regionalcalibration) for t ∈ yearlytime]
    oecdgrowth = first.(growthdata)
    oecdcumulative = [sum(oecdgrowth[year:(year + 10)]) for year in decadebegin]    
    
    rowgrowth = last.(growthdata)
    rowcumulative = [sum(rowgrowth[year:(year + 10)]) for year in decadebegin]
    
    oecdbar = @pgf Plot({fill=oecdcolor, color=oecdcolor, ybar_stacked}, Coordinates(decadeslabels, oecdcumulative))
    rowbar = @pgf Plot({fill=first(PALETTE), color=first(PALETTE), ybar_stacked}, Coordinates(decadeslabels, rowcumulative))

    push!(γfig, oecdbar, LegendEntry("OECD"), rowbar, LegendEntry("RoW"))
    
    if SAVEFIG
        for path in (PLOTPATH, THESISPATH)
            PGFPlotsX.save(joinpath(path, "growthmfig.tikz"), γfig; include_preamble=true)
        end
    end

    γfig
end

begin
    mbaumedian = timeseries_point_median(mbausims, yearlytime)
    mlower = timeseries_point_quantile(mbausims, 0.05, yearlytime)
    mupper = timeseries_point_quantile(mbausims, 0.95, yearlytime)

    mfig = @pgf Axis({
        width = raw"0.8\textwidth", height = raw"0.56\textwidth",
        grid = "both",
        ylabel = L"Business-as-usual\\carbon concentration $M_t^{b} \; [\si{\ppm}]$",
        ylabel_style = {align = "center"},
        xmin = 0, xmax = horizon,
        xtick = decadetime, xticklabels = decadeslabels,
        xticklabel_style = { rotate = 45 },
    })

    mediancoords = Coordinates(yearlytime, exp.(mbaumedian.u))
    medianplot = @pgf Plot({line_width = LINE_WIDTH}, mediancoords)
    markers = @pgf Plot({only_marks, mark_options = {scale = 1.5, draw_opacity = 0}, forget_plot, mark_repeat = 10}, mediancoords)

    lowerpath = @pgf Plot(
        {draw = "none", name_path = "lower", forget_plot}, 
        Coordinates(yearlytime, exp.(mlower.u)))

    upperpath = @pgf Plot(
        {draw = "none", name_path = "upper", forget_plot}, 
        Coordinates(yearlytime, exp.(mupper.u)))

    shading = @pgf Plot({opacity = 0.5, forget_plot}, "fill between [of=lower and upper]")

    push!(mfig, medianplot, markers, lowerpath, upperpath, shading)

    if SAVEFIG
        for path in (PLOTPATH, THESISPATH)
            PGFPlotsX.save(joinpath(path, "mbaufig.tikz"), mfig; include_preamble=true)
        end
    end

    mfig
end

# Economy
begin # Marginal abatement curve
    emissivity = range(0.0, 1.0; length = 51)
    timestyle = Dict(80 => "dashed", 0 => "solid")
    regioncolors = Dict(oecdeconomy => oecdcolor, roweconomy => first(PALETTE))

    xticks = 0:0.2:1
    xticklabels = [@sprintf("%.0f\\%%", 100 * x) for x in xticks]

    ytick = 0.02:0.02:0.12
    yticklabels = [@sprintf("%.f\\%%", 100 * y) for y in ytick]

    macfigure = @pgf Axis({
        width = raw"0.71\textwidth",
        height = raw"0.5\textwidth",
        grid = "both",
        xlabel = L"Abated percentage $\varepsilon(\alpha_t)$",
        ylabel = L"Abatement costs $\beta_t\big(\varepsilon(\alpha_t)\big)$",
        xmin = 0., xmax = 1.,
        xtick = xticks, xticklabels = xticklabels,
        ymin = 0., ymax = maximum(ytick),
        ytick = ytick, yticklabels = yticklabels,
        scaled_y_ticks = false,
        legend_style = { at = "{(0.3, 0.95)}" }
    })

    for (t, linestyle) in timestyle, (economy, color) in regioncolors
        mac = [β(t, ε, economy) for ε in emissivity]

        style = @pgf linestyle == "dashed" ? {dashed} : {}

        abatementcurve = @pgf Plot({line_width = LINE_WIDTH, color = color, style...}, Coordinates(emissivity, mac))

        push!(macfigure, abatementcurve, LegendEntry(@sprintf("%d", 2020 + t)))
    end

    if SAVEFIG
        for path in (PLOTPATH, THESISPATH)
            PGFPlotsX.save(joinpath(path, "macfigure.tikz"), macfigure; include_preamble=true)
        end
    end
    
    macfigure
end