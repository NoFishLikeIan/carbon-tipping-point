using Revise
using JLD2, CSV
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
    DATAPATH = "data"
    PLOTPATH = "plots"
    PRESENTATIONPATH = joinpath(PLOTPATH, "presentation")

    SAVEFIG = false
    LINE_WIDTH = 2.5
    SEED = 11148705

    TLABEL = L"Temperature $T_t \; [\si{\degree}]$"
    MLABEL = L"Carbon concentration $M_t \; [\si{\ppm}]$"
end;

begin # Construct models and grids
    thresholds = [1.5, 2.5]
    
    calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
    preferences = EpsteinZin()
    damages = GrowthDamages()
    economy = Economy()
    jump = Jump()
    hogg = Hogg()

    linearmodel = LinearModel(hogg, preferences, damages, economy, calibration)
    jumpmodel = JumpModel(jump, hogg, preferences, damages, economy, calibration)

    tippingmodels = TippingModel[]
    for Tᶜ ∈ thresholds
        albedo = Albedo(Tᶜ)
        model = TippingModel(albedo, hogg, preferences, damages, economy, calibration)

        push!(tippingmodels, model)
    end

    models = AbstractModel[tippingmodels..., linearmodel, jumpmodel]
    labels = ["Imminent", "Remote", "No Feedback", "Benchmark"]
    labelsbymodel = Dict(models .=> labels)
end;

begin # Labels, colors and axis
    PALETTE = colorschemes[:grays]
    colors = get(PALETTE, [0., 0.5, 0.7, 1])

    colorsbymodel = Dict(models .=> colors)
    ΔTmax = 6.
    ΔTspace = range(0.0, ΔTmax; length = 51)
    Tspace = ΔTspace .+ Hogg().Tᵖ

    horizon = round(Int64, calibration.tspan[2])
    yearlytime = 0:1:horizon

    temperatureticks = collect.(makedeviationtickz(0.0, ΔTmax, first(tippingmodels); step=1, digits=0))

    Tmin, Tmax = extrema(temperatureticks[1])

    m₀ = log(hogg.M₀)
    X₀ = [first(Model.Tstable(m₀, hogg)), m₀]
end;

begin # Load IPCC data
    IPCCDATAPATH = joinpath(DATAPATH, "climate-data", "proj-median.csv")
    ipccproj = CSV.read(IPCCDATAPATH, DataFrame)

    getscenario(s) = filter(:Scenario => isequal("SSP$(s) - Baseline"), ipccproj)

    bauscenario = getscenario(5)
end;

begin # Albedo plot
    albedovariation = [[Model.λ(T, model.hogg, model.albedo) for T in Tspace] for model in tippingmodels]

    ytick = 0.28:0.02:0.32

    albedofig = @pgf Axis(
        {
        width = raw"0.5\textwidth",
        height = raw"0.5\textwidth",
        grid = "both",
        xlabel = TLABEL,
        ylabel = "Positive feedback \$\\lambda(T_t)\$",
        xticklabels = temperatureticks[2],
        xtick = temperatureticks[1],
        xmin = Tmin, xmax = Tmax,
        ymin = ytick[1] - 0.01, ymax = ytick[end] + 0.01,
        ytick = ytick,
        legend_cell_align = "left"
    })

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "skeleton-albedo.tikz"), albedofig; include_preamble=true)
    end

    @pgf for (model, loss) in zip(tippingmodels, albedovariation)
        Tᶜ = model.albedo.Tᶜ
        curve = Plot({
                color = colorsbymodel[model], 
                line_width = LINE_WIDTH, opacity = 0.8
            }, Coordinates(zip(Tspace, loss)))

        label = labelsbymodel[model]
        legend = LegendEntry(label)

        push!(albedofig, curve, legend)
    end

    @pgf albedofig["legend style"] = raw"at = {(0.95, 0.95)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "albedo.tikz"), albedofig; include_preamble=true)
    end

    albedofig
end

begin
    simmodels = models[1:3]
    sims = Dict{AbstractModel, DiffEqArray}()

    for model in simmodels
        prob = SDEProblem(Fbau!, G!, X₀, (0.0, 200.0), model)
        sol = solve(EnsembleProblem(prob), trajectories = 1000)

        simpath = timeseries_point_quantile(sol, [0.2, 0.5, 0.8], yearlytime)

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
            M = Model.Mstable(T, model)
            isstable = model isa LinearModel || Model.radiativeforcing′(T, model.hogg, model.albedo) < 0
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

    Mmax = 1000.
    
    Mmedianpath = @. exp(getindex(getindex(simpath.u, 2), 2))
    Mticks = Mmedianpath[1:10:end]

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
        xlabel = L"Carbon concentration $M_t \; [\si{\ppm}]$ \\ and (year) reached in business-as-usual",
        yticklabels = temperatureticks[2],
        ytick = temperatureticks[1],
        ymin = Tmin, ymax = Tmax,
        legend_cell_align = "left",
        xmin = 380, xmax = 845,
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

        Tpath = @. getindex(simpath.u, 1)
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
        xmin = 380, xmax = 845,
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
    mbau(m, model, t) = γ(t, model.calibration)
    σₘbau(m, model, t) = model.hogg.σₘ

    mfn = SDEFunction(mbau, σₘbau)

    mbauprob = SDEProblem(mfn, log(hogg.M₀), (0.0, 80.0), first(tippingmodels))
    mensemble = EnsembleProblem(mbauprob)
    mbausims = solve(mensemble; trajectories=10_000)
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

    growthticks = (0.8:0.2:1.4) ./ 100

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

    medianplot = @pgf Plot({line_width = LINE_WIDTH}, Coordinates(zip(yearlytime, exp.(mbaumedian))))

    lowerplot = @pgf Plot({line_width = LINE_WIDTH, dotted, opacity = 0.5}, Coordinates(zip(yearlytime, exp.(mlower))))
    upperplot = @pgf Plot({line_width = LINE_WIDTH, dotted, opacity = 0.5}, Coordinates(zip(yearlytime, exp.(mupper))))

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
    m₀ = log(540)

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

            simulation = solve(densprob, SRIW1())
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
    denstemperatureticks = makedeviationtickz(0.0, ceil(Tmax - hogg.Tᵖ), first(tippingmodels); step=1, digits=0, addedlabels = [(L"$T_0$", T₀[1])])

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


function simulatebau(model::TippingModel; trajectories=100, X₀=[model.hogg.T₀, log(model.hogg.M₀)])
    prob = SDEProblem(Fbau!, G!, X₀, (0.0, 80.0), model)
    ensemble = EnsembleProblem(prob)

    sol = solve(ensemble; trajectories)

    return sol
end

function simulatebau(model::JumpModel; trajectories=100, X₀=[model.hogg.T₀, log(model.hogg.M₀)])
    diffprob = SDEProblem(Fbau!, G!, X₀, (0.0, 80.0), model)
    varjump = VariableRateJump(rate, affect!)
    prob = JumpProblem(diffprob, Direct(), varjump)

    ensemble = EnsembleProblem(prob)

    solve(ensemble, SRIW1(); trajectories)
end

bautime = 0:80

begin # Side by side BAU Δλ = 0.8
    baumodels = tippingmodels

    yeartickstep = 10

    baufig = @pgf GroupPlot(
        {
        group_style = {
            group_size = "1 by $(length(baumodels))",
            horizontal_sep = "1pt",
            xticklabels_at = "edge bottom"
        },
        width = raw"\textwidth",
        height = raw"0.6\textwidth",
        ymin = Tmin, ymax = Tmax,
        xmin = 0, xmax = maximum(bautime),
        xtick = 0:yeartickstep:maximum(bautime),
        xticklabels = (0:yeartickstep:maximum(bautime)) .+ BASELINE_YEAR,
        grid = "both", xticklabel_style = {rotate = 45}
    }
    )

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "baufig.tikz"), baufig; include_preamble=true)
    end

    for (i, model) in enumerate(baumodels)
        nextgroup = @pgf if i == 1
            {
                ylabel = TLABEL,
                yticklabels = temperatureticks[2],
                ytick = temperatureticks[1]
            }
        elseif i == length(baumodels)
            {
                ylabel = TLABEL,
                xlabel = "Year",
                yticklabels = temperatureticks[2][1:(end-1)],
                ytick = temperatureticks[1][1:(end-1)]
            }
        else
            {
                ylabel = TLABEL,
                yticklabels = temperatureticks[2][1:(end-1)],
                ytick = temperatureticks[1][1:(end-1)]
            }
        end

        push!(baufig, nextgroup)

        bausim = simulatebau(model; trajectories=10_000)
        medianpath = first.(timeseries_point_median(bausim, bautime).u)
        lower = first.(timeseries_point_quantile(bausim, 0.05, bautime).u)
        upper = first.(timeseries_point_quantile(bausim, 0.95, bautime).u)

        medianplot = @pgf Plot({line_width = LINE_WIDTH}, Coordinates(zip(bautime, medianpath)))

        lowerplot = @pgf Plot({draw = "none", "name path=lower"}, Coordinates(zip(bautime, lower)))
        upperplot = @pgf Plot({draw = "none", "name path=upper"}, Coordinates(zip(bautime, upper)))
        confidencebands = @pgf Plot({fill = "gray", opacity = 0.4}, raw"fill between [of=lower and upper]")

        push!(baufig, lowerplot, upperplot, confidencebands, medianplot)

    end

    @pgf baufig["legend style"] = raw"at = {(0.45, 0.95)}"


    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "baufig.tikz"), baufig; include_preamble=true)
    end

    baufig
end

begin # Carbon decay calibration
    tippingmodel = last(tippingmodels)

    sinkspace = range(tippingmodel.hogg.N₀, 1.2 * tippingmodel.hogg.N₀; length=51)
    decay = [Model.δₘ(n, tippingmodel.hogg) for n in sinkspace]

    decayfig = @pgf Axis(
        {
        width = raw"0.5\textwidth",
        height = raw"0.5\textwidth",
        grid = "both",
        xlabel = raw"Carbon stored in sinks $N$",
        ylabel = raw"Decay of CO$_2$ in the atmosphere $\delta_m$",
        xmin = first(sinkspace), xmax = last(sinkspace),
        ymin = 0
    }
    )

    @pgf push!(decayfig, Plot({line_width = LINE_WIDTH}, Coordinates(zip(sinkspace, decay))))


    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "decay.tikz"), decayfig; include_preamble=true)
    end

    decayfig
end

begin # Carbon decay path
    bausim = simulatebau(tippingmodel)
    bauM = exp.(getindex.(timeseries_point_median(bausim, bautime).u, 2))
    mediandecay = [Model.δₘ(M, tippingmodel.hogg) for M in bauM]

    decaypathfig = @pgf Axis({
        width = raw"0.7\textwidth",
        height = raw"0.5\textwidth",
        grid = "both",
        xlabel = raw"Carbon concentration $M$",
        ylabel = raw"Decay of CO$_2$ in the atmosphere $\delta_m$",
        xmin = tippingmodel.hogg.M₀, xmax = 800,
        scaled_y_ticks = false
    })

    @pgf push!(decaypathfig,
        Plot({line_width = LINE_WIDTH}, Coordinates(bauM, mediandecay))
    )


    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "decaypathfig.tikz"), decaypathfig; include_preamble=true)
    end

    decaypathfig
end

begin # Damage fig
    ds = [Model.d(T, damages, hogg) for T in Tspace]

    maxpercentage = ceil(maximum(ds), digits=2)
    ytick = 0:0.02:maxpercentage
    yticklabels = [@sprintf("%.0f \\%%", 100 * y) for y in ytick]

    damagefig = @pgf Axis({
        width = raw"0.5\textwidth",
        height = raw"0.5\textwidth",
        grid = "both",
        xlabel = TLABEL,
        ylabel = raw"Damage function $d(T_t)$",
        xticklabels = temperatureticks[2],
        xtick = temperatureticks[1],
        xmin = Tmin, xmax = maximum(Tspace),
        xticklabel_style = {rotate = 45},
        yticklabels = yticklabels, ytick = ytick, ymin = 0.,
        scaled_y_ticks = false,
    })

    @pgf damagecurve = Plot({line_width = LINE_WIDTH},
        Coordinates(Tspace, ds)
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