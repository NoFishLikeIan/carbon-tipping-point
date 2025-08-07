using Revise

using FastClosures
using JLD2, CSV, UnPack
using DataFrames, DataStructures
using StatsBase
using Interpolations

using DifferentialEquations, DifferentialEquations.EnsembleAnalysis

using Plots, Printf, PGFPlotsX, Colors, ColorSchemes, LaTeXStrings
using Statistics

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{siunitx}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\DeclareSIUnit{\ppm}{p.p.m.}")

using Model, Grid

includet("../../../src/extensions.jl")
includet("../../utils/saving.jl")
includet("../../utils/simulating.jl")
includet("../utils.jl")

begin # Global variables
    DATAPATH = "data"
    PLOTPATH = "papers/job-market-paper/submission/plots"
    PRESENTATIONPATH = joinpath(PLOTPATH, "presentation")

    SAVEFIG = false
    LINE_WIDTH = 2.5
    SEED = 11148705

    trajectories = 10_000

    TLABEL = L"Temperature $T_t \; [\si{\degree}]$"
    MLABEL = L"\si{CO2}e $M_t \; [\si{\ppm}]$"
end;

begin # Construct models and grids
    calibrationpath = joinpath(DATAPATH, "calibration.jld2")
    @assert isfile(calibrationpath) "Calibration file not found at $calibrationpath"
    
    calibrationfile = jldopen(calibrationpath, "r+")
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = calibrationfile
    close(calibrationfile)

    preferences = EpsteinZin()
    damages = GrowthDamages()
    economy = Economy()
    jump = Jump()

    linearmodel = LinearModel(hogg, preferences, damages, economy)
    jumpmodel = JumpModel(hogg, preferences, damages, economy, jump)

    tippingmodels = [
        TippingModel(hogg, preferences, damages, economy, Model.updateTᶜ(2.5 + hogg.Tᵖ, feedbackhigher)),
        TippingModel(hogg, preferences, damages, economy, Model.updateTᶜ(feedbackhigher.Tᶜ, feedbacklower))
    ]

    models = AbstractModel[tippingmodels..., linearmodel, jumpmodel]
    labels = ["Imminent", "Remote", "No Feedback", "Benchmark"]
    labelsbymodel = Dict(models .=> labels)
end;

begin # Labels, colors and axis
    PALETTE = colorschemes[:grays]
    colors = get(PALETTE, range(0., 1.; length = length(models)), (0., 1.))

    colorsbymodel = Dict(models .=> colors)
    ΔTmax = 6.5
    ΔTspace = range(0.0, ΔTmax; length = 101)
    Tspace = ΔTspace .+ Hogg().Tᵖ

    horizon = 2100. - calibration.baselineyear
    yearlytime = 0:1:horizon
    simtspan = (0, horizon)

    temperatureticks = collect.(makedeviationtickz(1., ΔTmax, first(tippingmodels); step=1, digits=0))

    Tmin, Tmax = extrema(temperatureticks[1])

    m₀ = log(hogg.M₀ / hogg.Mᵖ)
    X₀ = [hogg.T₀, m₀]
end;

begin # Feedback plot
    additionalradiation = [λ(T, model.feedback) for T in Tspace, model in tippingmodels]

    feedbackfig = @pgf Axis({
        width = raw"0.6\textwidth",
        height = raw"0.5\textwidth",
        grid = "both",
        xlabel = TLABEL,
        ylabel = raw"Positive feedback $\lambda(T_t) \; [\si{W.m^{-2}}]$",
        xticklabels = temperatureticks[2],
        xtick = temperatureticks[1],
        xmin = Tmin, xmax = Tmax,
        legend_cell_align = "left",
        legend_style = { at = {"(0.025, 0.975)"}, anchor = "north west", nodes = {scale = 0.7} }
    })

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "skeleton-albedo.tikz"), feedbackfig; include_preamble=true)
    end

    for mdx in axes(additionalradiation, 2)
        rad = @view additionalradiation[:, mdx]
        model = tippingmodels[mdx]
        
        curve = @pgf Plot({ color = colorsbymodel[model], line_width = LINE_WIDTH, opacity = 0.8 }, Coordinates(Tspace, rad))

        push!(feedbackfig, curve, LegendEntry(labelsbymodel[model]))
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "feedbackfig.tikz"), feedbackfig; include_preamble=true)
    end

    feedbackfig
end

begin
    simmodels = models[1:3]
    sims = Dict{AbstractModel, DiffEqArray}()

    npprob = SDEProblem(Fnp!, G!, X₀, simtspan, (linearmodel, calibration)) |> EnsembleProblem

    for model in simmodels
        npparameters = (model, calibration)
        sol = solve(npprob, SOSRI(); trajectories, p = npparameters)

        @printf "Done with simulation of %s\n" labelsbymodel[model]

        simpath = timeseries_point_quantile(sol, (0.05, 0.5, 0.95), yearlytime)

        sims[model] = simpath
    end
end;

begin # Nullcline plot
    nullclinevariation = Dict{AbstractModel, Vector{Vector{NTuple{2,Float64}}}}()

    for model in reverse(simmodels)
        nullclines = Vector{NTuple{2,Float64}}[]
        currentM = NTuple{2,Float64}[]
        currentlystable = true

        for T in Tspace
            m = mstable(T, model)
            M = hogg.Mᵖ * exp(m)
            isstable = model isa LinearModel || ∂μ∂T(T, m, model.hogg, model.feedback) < 0
            if isstable == currentlystable
                push!(currentM, (M, T - 0.1))
            else
                currentlystable = !currentlystable
                push!(nullclines, currentM)
                currentM = [(M, T - 0.1)]
            end
        end

        push!(nullclines, currentM)
        nullclinevariation[model] = nullclines
    end

    
    mmedianpath = getindex.(getindex.(sims[simmodels[1]].u, 2), 2)
    Mmedianpath = @. hogg.Mᵖ * exp(mmedianpath)
    Mticks = Mmedianpath[1:15:end]
    Mmin, Mmax = extrema(Mticks)

    Mtickslabels = [
        L"\small $%$M$\\ \footnotesize ($%$y$)"
        for (M, y) in zip(round.(Int, Mticks), 2020:10:2100)
    ]

    nullclinefig = @pgf Axis({
        width = raw"0.9\textwidth",
        height = raw"0.7\textwidth",
        grid = "both",
        ylabel = TLABEL,
        xlabel_style = {align = "center"},
        xlabel = L"\si{CO2}e concentration $M_t \; [\si{\ppm}]$ \\ and (year) reached in the no-policy scenario",
        yticklabels = temperatureticks[2],
        ytick = temperatureticks[1],
        ymin = Tmin, ymax = Tmax,
        legend_cell_align = "left",
        xmin = floor(Mmin, digits = -2), 
        xmax = ceil(Mmax, digits = -2),
        xtick = Mticks, xticklabels = Mtickslabels,
        xticklabel_style = {align = "center"}
    })

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "skeleton-nullcline.tikz"), nullclinefig; include_preamble=true)
    end

    for model in reverse(simmodels) # Nullclines
        color = colorsbymodel[model]
        
        stableleft, rest... = nullclinevariation[model]

        leftcurve = @pgf Plot({color = color, line_width = LINE_WIDTH / 2, forget_plot}, Coordinates(stableleft))

        push!(nullclinefig, leftcurve)

        if !isempty(rest)
            unstable, stableright = rest
            unstablecurve = @pgf Plot({color = color, line_width = LINE_WIDTH / 2, forget_plot, dotted}, Coordinates(unstable))
            rightcurve = @pgf Plot({color = color, line_width = LINE_WIDTH / 2, forget_plot}, Coordinates(stableright))

            push!(nullclinefig, unstablecurve, rightcurve)
        end
    end

    for model in reverse(simmodels) # Simulation plots
        color = colorsbymodel[model]
        simpath = sims[model]

        Tpath = getindex.(simpath.u, 1)
        lower, median, upper = (getindex.(Tpath, i) for i in 1:3)

        mediancoords = Coordinates(Mmedianpath, median)
        curve = @pgf Plot({color = color, line_width = LINE_WIDTH}, mediancoords)

        markers = @pgf Plot({only_marks, mark_options = {fill = "black", scale = 1.5, draw_opacity = 0, color = color}, mark_repeat = 10, forget_plot}, mediancoords)

        label = labelsbymodel[model]
        legend = LegendEntry(label)

        # Add shading between lower and upper curves
        lowerpath = @pgf Plot({draw = "none", name_path = "lower", forget_plot}, Coordinates(Mmedianpath, lower))
        upperpath = @pgf Plot({draw = "none", name_path = "upper", forget_plot}, Coordinates(Mmedianpath, upper))
        shading = @pgf Plot({fill = color, opacity = 0.15, forget_plot}, raw"fill between [of=lower and upper]")

        push!(nullclinefig, curve, legend, markers, lowerpath, upperpath, shading)
    end

    @pgf nullclinefig["legend style"] = raw"at = {(0.95, 0.3)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "nullcline.tikz"), nullclinefig; include_preamble=true)
    end

    nullclinefig
end

begin # Pure nullcline figure
    nullclinefig = @pgf Axis({
        width = raw"0.9\textwidth",
        height = raw"0.7\textwidth",
        grid = "both",
        ylabel = TLABEL,
        xlabel_style = {align = "center"},
        xlabel = MLABEL,
        yticklabels = temperatureticks[2],
        ytick = temperatureticks[1],
        ymin = Tmin, ymax = Tmax,
        xmin = floor(Mmin, digits = -2), 
        xmax = ceil(Mmax, digits = -2),
        legend_cell_align = "left"
    })

    for model in reverse(simmodels) # Nullcline plots
        color = colorsbymodel[model]
        
        stableleft, rest... = nullclinevariation[model]

        leftcurve = @pgf Plot({color = color, line_width = LINE_WIDTH}, Coordinates(stableleft))

        label = LegendEntry(labelsbymodel[model])

        push!(nullclinefig, leftcurve, label)

        if !isempty(rest)
            unstable, stableright = rest
            unstablecurve = @pgf Plot({color = color, line_width = LINE_WIDTH, forget_plot, dotted}, Coordinates(unstable))
            rightcurve = @pgf Plot({color = color, line_width = LINE_WIDTH, forget_plot}, Coordinates(stableright))

            push!(nullclinefig, unstablecurve, rightcurve)
        end
    end

    @pgf nullclinefig["legend style"] = raw"at = {(0.95, 0.3)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "just-nullclines.tikz"), nullclinefig; include_preamble=true)
    end

    nullclinefig
end

# --- Business-as-usual dynamics
begin # Simulate carbon concentrations
    mbau(m, parameters, t) = γ(t, parameters[2])
    σₘbau(m, parameters, t) = parameters[1].σₘ
    mfn = SDEFunction(mbau, σₘbau)
    
    parameters = (hogg, calibration)
    mbauprob = SDEProblem(mfn, log(hogg.M₀ / hogg.Mᵖ), (0.0, 80.0), parameters)
    mensemble = EnsembleProblem(mbauprob)
    mbausims = solve(mensemble, SOSRI(); trajectories = 10_000)
end

begin # Growth of carbon concentration 
    horizon = Int(last(yearlytime))

    figsize = @pgf {
        width = raw"0.425\linewidth",
        height = raw"0.3\linewidth",
    }

    gfig = @pgf GroupPlot({
        group_style = {
            group_size = "2 by 1",
            horizontal_sep = raw"0.15\linewidth"
        },
        xmin = 0.0, xmax = horizon
    })

    growthticks = (0.4:0.2:1.4) ./ 100

    γfig = @pgf Axis({})

    gdata = [γ(t, calibration) for t ∈ yearlytime]
    coords = Coordinates(zip(yearlytime, gdata))

    markers = @pgf Plot({only_marks, mark_options = {fill = "black", scale = 1.5, draw_opacity = 0}, mark_repeat = 10}, coords)

    curve = @pgf Plot({color = "black", line_width = "0.1cm"}, coords)

    ymin, ymax = extrema(growthticks)
    xtick = 0:20:horizon
    xticklabels = ["\\footnotesize $(Int(y + 2020))" for y in xtick]

    @pgf push!(gfig, {
            figsize...,
            grid = "both",
            ylabel = raw"\footnotesize Growth rate $\gamma_t^{b}$",
            ytick = growthticks,
            ymin = ymin,
            ymax = ymax,
            yticklabels = [@sprintf("\\footnotesize %.1f\\%%", 100 * x) for x in growthticks],
            xtick = xtick,
            xmin = 0, xmax = horizon,
            xticklabels = xticklabels,
            xticklabel_style = {rotate = 45},
            scaled_y_ticks = false
        }, curve, markers)


    mbaumedian = timeseries_point_median(mbausims, yearlytime)
    mlower = timeseries_point_quantile(mbausims, 0.05, yearlytime)
    mupper = timeseries_point_quantile(mbausims, 0.95, yearlytime)

    mfig = Axis()

    medianplot = @pgf Plot({line_width = LINE_WIDTH}, Coordinates(yearlytime, hogg.Mᵖ .* exp.(mbaumedian.u)))

    lowerplot = @pgf Plot({line_width = LINE_WIDTH, dotted, opacity = 0.5}, Coordinates(yearlytime, hogg.Mᵖ .* exp.(mlower.u)))
    upperplot = @pgf Plot({line_width = LINE_WIDTH, dotted, opacity = 0.5}, Coordinates(yearlytime, hogg.Mᵖ .* exp.(mupper.u)))

    push!(mfig, medianplot, lowerplot, upperplot)

    @pgf push!(gfig, {
            figsize...,
            grid = "both",
            ylabel = L"\footnotesize \\$M_t^{b} \; [\si{\ppm}]$",
            ylabel_style = {align = "center"},
            xtick = xtick,
            xmin = 0, xmax = horizon,
            xticklabels = xticklabels,
            xticklabel_style = {rotate = 45},
        }, mfig)

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "growthmfig.tikz"), gfig; include_preamble=true)
    end

    gfig
end

# TODO: Finish the comparison of density plots between jump and tipping models.
densmodels = AbstractModel[last(tippingmodels), jumpmodel];
begin # Simulates the two models    
    temperaturedrift! = @closure (du, u, model, t) -> begin
        du[1] = μ(u[1], m₀, model) / model.hogg.ϵ
    end

    temperaturenoise! = @closure (Σ, u, model, t) -> begin
        Σ[1] = model.hogg.σₜ / model.hogg.ϵ
    end

    T₀ = minimum(Model.Tstable(m₀, jumpmodel))

    simulations = Dict{AbstractModel, RODESolution}()
    
    for model in densmodels
        isjump = model isa JumpModel
        fn = SDEFunction(temperaturedrift!, temperaturenoise!)

        densprob = SDEProblem(fn, [T₀], (0., 50_000.), model)
        
        if !isjump
            simulation = solve(densprob)
        else
            onedrate(u, model, t) = intensity(u[1], model.hogg, model.jump)
            
            function onedtipping!(integrator)
                model = integrator.p
                q = increase(integrator.u[1], model.hogg, model.jump)
                integrator.u += q
            end

            ratejump = VariableRateJump(onedrate, onedtipping!);
            jumpprob = JumpProblem(densprob, ratejump)

            simulation = solve(densprob, SOSRI())
        end
        
        simulation = solve(densprob)
        simulations[model] = simulation
    end
end
begin # Construct histograms
    Textrema = map(sim -> extrema(first.(sim.u)), values(simulations))
    Thistmin = minimum(first.(Textrema)) - 0.1
    Thistmax = maximum(last.(Textrema)) + 0.1

    Tbins = range(Thistmin, Thistmax; length = 31)

    histograms = Dict{AbstractModel, Histogram}()

    for model in densmodels
        simulation = simulations[model]
        T = first.(simulation.u)

        histogram = fit(Histogram, T, Tbins)
        histograms[model] = histogram
    end
end
begin
    denstemperatureticks = makedeviationtickz(0.0, ceil(Tmax - hogg.Tᵖ), first(tippingmodels); step=1, digits=0, addedlabels = [(L"$T_0$", hogg.T₀)])

    densfig = @pgf GroupPlot({
        group_style = {
            group_size = "1 by 2",
            xticklabels_at = "edge bottom",
            vertical_sep = "2pt"
        }
    })
    

    for (k, model) in enumerate(densmodels)
        histogram = histograms[model]

        densityplot = @pgf Plot({
            "ybar interval",
            "xticklabel interval boundaries",
            xmajorgrids = false,
            ylabel = labelsbymodel[model],
            fill = "gray", opacity = 0.5
        }, Table(histogram))


        @pgf lastplotopt = k < length(densmodels) ? {} : { xlabel = TLABEL, xticklabels = denstemperatureticks[2] }

        @pgf push!(densfig, {lastplotopt...,
            xmin = Thistmin, xmax = Thistmax, 
            xtick = denstemperatureticks[1],
            ymin = 0, ymax = 5000,
            grid = "both", yticklabels = raw"\empty",
            width = raw"0.7\textwidth", height = raw"0.4\textwidth",
            ylabel = "$(labelsbymodel[model])"
        }, densityplot)
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "densfig.tikz"), densfig; include_preamble=true)
    end

    densfig
end

begin # Carbon decay calibration
    sinkspace = range(hogg.N₀, 2hogg.N₀; length=51)
    decay = [Model.δₘ(n, hogg) for n in sinkspace]

    decayfig = @pgf Axis({
        width = raw"0.5\textwidth",
        height = raw"0.5\textwidth",
        grid = "both",
        xlabel = raw"Carbon stored in sinks $N$",
        ylabel = raw"Decay of CO$_2$ in the atmosphere $\delta_m$",
        xmin = first(sinkspace), xmax = last(sinkspace),
        ymin = 0
    })

    @pgf push!(decayfig, Plot({line_width = LINE_WIDTH}, Coordinates(sinkspace, decay)))

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "decay.tikz"), decayfig; include_preamble=true)
    end

    decayfig
end

begin # Carbon decay path
    npmprob = ODEProblem((m, calibration, t) -> γ(t, calibration), m₀, simtspan, calibration)

    npsim = solve(npmprob, AutoVern9(Rodas5P()); saveat = 1.)
    npM = hogg.Mᵖ * exp.(npsim.u)
    mediandecay = δₘ.(npM, hogg)

    decaypathfig = @pgf Axis({
        width = raw"0.7\textwidth",
        height = raw"0.5\textwidth",
        grid = "both",
        xlabel = raw"Carbon concentration $M$",
        ylabel = raw"Decay of CO$_2$ in the atmosphere $\delta_m$",
        xmin = hogg.M₀, xmax = 800.,
        scaled_y_ticks = false
    })

    @pgf push!(decaypathfig,
        Plot({line_width = LINE_WIDTH}, Coordinates(npM, mediandecay))
    )

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "decaypathfig.tikz"), decaypathfig; include_preamble=true)
    end

    decaypathfig
end

begin # Damage fig
    ΔTspace = range(0, ΔTmax; length = 51)
    cumulativedamages = Model.D.(ΔTspace, damages)

    maxpercentage = ceil(maximum(cumulativedamages), digits=2)
    ytick = 0:0.05:maxpercentage
    yticklabels = [@sprintf("%.0f \\%%", 100 * y) for y in ytick]

    _, xticklabels = makedeviationtickz(ΔTspace[1], ΔTspace[end], first(tippingmodels); step = 1, digits = 0)
    xtick = ΔTspace[1]:1:ΔTspace[end]

    damagefig = @pgf Axis({
        width = raw"0.5\textwidth",
        height = raw"0.5\textwidth",
        grid = "both",
        xlabel = TLABEL,
        ylabel = raw"Cumulative damages $D(T_t)$",
        xmin = 0, xmax = ΔTmax,
        xticklabel_style = {rotate = 45},
        yticklabels = yticklabels, ytick = ytick, ymin = 0.,
        xticklabels = xticklabels, xtick = xtick,
        scaled_y_ticks = false,
    })

    @pgf damagecurve = Plot({line_width = LINE_WIDTH},
        Coordinates(ΔTspace, cumulativedamages)
    )

    push!(damagefig, damagecurve)

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "damagefig.tikz"), damagefig; include_preamble=true)
    end

    damagefig
end

begin # Marginal abatement curve
    emissivity = range(0.0, 1.0; length = 51)

    times = [0., 40., 80.] |> reverse

    yearcolors = get(PALETTE, [0., 0.4, 0.6]) # graypalette(length(times))

    xticks = 0:0.2:1
    xticklabels = [@sprintf("%.0f\\%%", 100 * x) for x in xticks]

    ytick = 0.02:0.02:0.12
    yticklabels = [@sprintf("%.f\\%%", 100 * y) for y in ytick]

    abatementfig = @pgf Axis({
        width = raw"0.71\textwidth",
        height = raw"0.5\textwidth",
        grid = "both",
        xlabel = L"Abated percentage $\varepsilon(\alpha_t)$",
        ylabel = L"Abatement costs $\beta_t\big(\varepsilon(\alpha_t)\big)$",
        xmin = 0., xmax = 1.,
        xtick = xticks, xticklabels = xticklabels,
        ymin = 0., ymax = maximum(ytick),
        ytick = ytick, yticklabels = yticklabels,
        scaled_y_ticks = false
    })

    for (k, t) in enumerate(times)
        mac = [β(t, ε, economy) for ε in emissivity]

        abatementcurve = @pgf Plot({line_width = LINE_WIDTH, color = yearcolors[k]}, Coordinates(emissivity, mac))

        push!(abatementfig, abatementcurve, LegendEntry(@sprintf("%d", 2020 + t)))
    end

    @pgf abatementfig["legend style"] = raw"at = {(0.3, 0.95)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "abatementfig.tikz"), abatementfig; include_preamble=true)
    end
    
    abatementfig
end