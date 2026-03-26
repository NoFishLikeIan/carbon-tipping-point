using Revise

using FastClosures
using JLD2, CSV, UnPack
using DataFrames, DataStructures
using StatsBase
using Interpolations
using StaticArrays
using Printf

using DifferentialEquations, DifferentialEquations.EnsembleAnalysis
using Roots

using Plots, Printf, PGFPlotsX, Colors, ColorSchemes, LaTeXStrings
using Statistics

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{siunitx}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\DeclareSIUnit{\ppm}{p.p.m.}")

using Model, Grid

includet("../utils.jl")

includet("../../../src/valuefunction.jl")
includet("../../../src/regret.jl")
includet("../../../src/extend/model.jl")

includet("../../utils/simulating.jl")
includet("../../utils/saving.jl")
includet("../../utils/simulating.jl")

begin # Global variables
    DATAPATH = "data"
    PLOTPATH = "../job-market-paper/jeem/plots"
    PRESENTATIONPATH = joinpath(PLOTPATH, "presentation")

    SAVEFIG = true
    LINE_WIDTH = 2.5
    SEED = 11148705

    trajectories = 10_000

    TLABEL = L"Temperature $T_t \; [\si{\degree}]$"
    MLABEL = L"\si{CO2}e $M_t \; [\si{\ppm}]$"
end;

begin # Construct models and grids
    calibrationpath = joinpath(DATAPATH, "calibration")

    climatepath = joinpath(calibrationpath, "climate.jld2")
    @assert isfile(climatepath) "Climate calibration file not found at $climatepath"
    climatefile = jldopen(climatepath, "r+")
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher, decay = climatefile
    close(climatefile)

    abatementpath = joinpath(calibrationpath, "abatement.jld2")
    @assert isfile(abatementpath) "Abatement calibration file not found at $abatementpath"
    abatementfile = jldopen(abatementpath, "r+")
    @unpack abatement = abatementfile
    close(abatementfile)

    investments = Investment()
    damages = BurkeHsiangMiguel() # NoDamageGrowth{Float64}()
    economy = Economy(investments = investments, damages = damages, abatement = abatement)
    
    preferences = LogSeparable()

    linearmodel = IAM(LinearClimate(hogg, decay), economy, preferences)
    
    tippingmodels = [
        IAM(TippingClimate(hogg, decay, updatethreshold(2., feedback)), economy, preferences),
        IAM(TippingClimate(hogg, decay, updatethreshold(4., feedback)), economy, preferences)
    ]

    models = IAM[tippingmodels..., linearmodel]
    labels = [L"T^c = 2\si{\degree}", L"T^c = 4\si{\degree}", "Linear"]
    labelsbymodel = Dict(models .=> labels)
end;

begin # Labels, colors and axis
    PALETTE = colorschemes[:grays]
    colors = get(PALETTE, range(0., 1.; length = length(models)), (0., 1.25))
    trajectorymarkers = ["square*", "diamond*", "*"]

    colorsbymodel = Dict(models .=> colors)
    markerbymodel = Dict(models .=> trajectorymarkers)
    Tmin = 0.0; Tmax = 6.0
    Tspace = range(Tmin, Tmax; length = 101)

    horizon = 2100. - first(calibration.calibrationspan)
    yearlytime = 0:1:horizon
    simtspan = (0, horizon)

    temperatureticks = collect.(makedeviationtickz(0, 6; step=1, digits=0))

    m₀ = log(hogg.M₀ / hogg.Mᵖ)
    T₀ = hogg.T₀
    X₀ = SVector(T₀, m₀)
end;

begin # Feedback plot
    additionalradiation = [model.climate isa TippingClimate ? λ(T, model.climate.feedback) : 0. for T in Tspace, model in models]

    feedbackfig = @pgf Axis({
        width = raw"0.51\textwidth",
        height = raw"0.425\textwidth",
        grid = "both",
        xlabel = TLABEL,
        ylabel = raw"Positive feedback $\lambda(T_t) \; [\si{W.m^{-2}}]$",
        xticklabels = temperatureticks[2],
        xtick = temperatureticks[1],
        xmin = Tmin, xmax = Tmax,
        ymax = 3.,
        legend_cell_align = "left",
        legend_style = { at = {"(0.025, 0.975)"}, anchor = "north west", nodes = {scale = 0.7} }
    })

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "skeleton-albedo.tikz"), feedbackfig; include_preamble=true)
    end

    for mdx in reverse(axes(additionalradiation, 2))
        rad = @view additionalradiation[:, mdx]
        model = models[mdx]
        
        radiationcurve = @pgf Plot({ color = colorsbymodel[model], line_width = LINE_WIDTH, opacity = 0.8 }, Coordinates(Tspace, rad))

        push!(feedbackfig, radiationcurve, LegendEntry(labelsbymodel[model]))
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "feedbackfig.tikz"), feedbackfig; include_preamble=true)
    end

    feedbackfig
end


begin # Equilibria figure
    additionalradiation = [model.climate isa TippingClimate ? λ(T, model.climate.feedback) : 0. for T in Tspace, model in models]

    feedbackfig = @pgf Axis({
        width = raw"0.51\textwidth",
        height = raw"0.425\textwidth",
        grid = "both",
        xlabel = TLABEL,
        ylabel = raw"Positive feedback $\lambda(T_t) \; [\si{W.m^{-2}}]$",
        xticklabels = temperatureticks[2],
        xtick = temperatureticks[1],
        xmin = Tmin, xmax = Tmax,
        ymax = 3.,
        legend_cell_align = "left",
        legend_style = { at = {"(0.025, 0.975)"}, anchor = "north west", nodes = {scale = 0.7} }
    })

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "skeleton-albedo.tikz"), feedbackfig; include_preamble=true)
    end

    for mdx in reverse(axes(additionalradiation, 2))
        rad = @view additionalradiation[:, mdx]
        model = models[mdx]
        
        radiationcurve = @pgf Plot({ color = colorsbymodel[model], line_width = LINE_WIDTH, opacity = 0.8 }, Coordinates(Tspace, rad))

        push!(feedbackfig, radiationcurve, LegendEntry(labelsbymodel[model]))
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "feedbackfig.tikz"), feedbackfig; include_preamble=true)
    end

    feedbackfig
end

begin # Simulate NP problem
    sims = Dict{IAM, DiffEqArray}()

    npprob = SDEProblem(Fnp, noise, X₀, simtspan, (linearmodel, calibration))
    npensemble = EnsembleProblem(npprob)

    for model in models
        npparameters = (model, calibration)
        sol = solve(npensemble; reltol = 1e-8, trajectories = 15_000, p = npparameters, saveat = 1.0)

        @printf "Done with simulation of %s\n" labelsbymodel[model]

        simpath = timestep_quantile(sol, (0.05, 0.5, 0.95), :)
        sims[model] = simpath
    end
end;

begin # NP simulation + nullclines
    nullclinevariation = Dict{IAM, Vector{Vector{NTuple{2,Float64}}}}()

    for model in reverse(models)
        nullclines = Vector{NTuple{2,Float64}}[]
        currentM = NTuple{2,Float64}[]
        currentlystable = true

        for T in Tspace
            m = mstable(T, model.climate)
            M = hogg.Mᵖ * exp(m)
            isstable = model.climate isa LinearClimate || ∂μ∂T(T, model.climate) < 0
            if isstable == currentlystable
                push!(currentM, (M, T))
            else
                currentlystable = !currentlystable
                push!(nullclines, currentM)
                currentM = [(M, T)]
            end
        end

        push!(nullclines, currentM)
        nullclinevariation[model] = nullclines
    end
    
    mmedianpath = getindex.(getindex.(sims[models[1]].u, 2), 2)
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

    for model in reverse(models) # Nullclines
        color = colorsbymodel[model]
        
        stableleft, rest... = nullclinevariation[model]

        leftcurve = @pgf Plot({color = color, line_width = LINE_WIDTH / 4, forget_plot}, Coordinates(stableleft))

        push!(nullclinefig, leftcurve)

        if !isempty(rest)
            unstable, stableright = rest
            unstablecurve = @pgf Plot({color = color, line_width = LINE_WIDTH / 4, forget_plot, dotted}, Coordinates(unstable))
            rightcurve = @pgf Plot({color = color, line_width = LINE_WIDTH / 4, forget_plot}, Coordinates(stableright))

            push!(nullclinefig, unstablecurve, rightcurve)
        end
    end

    for model in reverse(models) # Simulation plots
        color = colorsbymodel[model]
        marker = markerbymodel[model]
        simpath = sims[model]

        Tpath = getindex.(simpath.u, 1)
        lower, median, upper = (getindex.(Tpath, i) for i in 1:3)

        mediancoords = Coordinates(Mmedianpath, median)
        curve = @pgf Plot({color = color, line_width = LINE_WIDTH, forget_plot}, mediancoords)

        markers = @pgf Plot({only_marks, mark_options = {fill = "black", scale = 1.5, draw_opacity = 0, color = color, mark = marker}, mark_repeat = 10}, mediancoords)

        label = labelsbymodel[model]
        legend = LegendEntry(label)

        # Add shading between lower and upper curves
        lowerpath = @pgf Plot({draw = "none", name_path = "lower", forget_plot}, Coordinates(Mmedianpath, lower))
        upperpath = @pgf Plot({draw = "none", name_path = "upper", forget_plot}, Coordinates(Mmedianpath, upper))
        shading = @pgf Plot({fill = color, opacity = 0.05, forget_plot}, raw"fill between [of=lower and upper]")

        push!(nullclinefig, curve, markers, legend, lowerpath, upperpath, shading)
    end

    @pgf nullclinefig["legend style"] = raw"at = {(0.95, 0.3)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "nullcline.tikz"), nullclinefig; include_preamble=true)
    end

    nullclinefig
end

begin # Pure nullcline figure
    nullclinefig = @pgf Axis({
        width = raw"0.765\textwidth",
        height = raw"0.595\textwidth",
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

    for model in reverse(models) # Nullcline plots
        color = colorsbymodel[model]
        marker = markerbymodel[model]
        
        stableleft, rest... = nullclinevariation[model]

        leftcurve = @pgf Plot({
            color = color,
            line_width = LINE_WIDTH,
            mark = marker,
            mark_repeat = 10,
            mark_options = {fill = color, scale = 1.2}
        }, Coordinates(stableleft))

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

# --- No Policy dynamics
begin # Simulate carbon concentrations
    mbau(m, (hogg, calibration), t) = γ(t, calibration)
    parameters = (hogg, calibration)
    mnpproblem = ODEProblem(mbau, m₀, (0.0, 80.0), parameters)
    mnptraj = solve(mnpproblem, AutoVern9(Rodas5P()); saveat = 1.)
end

begin # Growth of carbon concentration 
    figsize = @pgf {
        width = raw"0.361\linewidth",
        height = raw"0.255\linewidth",
    }

    gfig = @pgf GroupPlot({
        group_style = {
            group_size = "2 by 1",
            horizontal_sep = raw"0.2\textwidth"
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

    mfig = Axis()

    Mpath = @. exp(mnptraj.u) * hogg.Mᵖ
    medianplot = @pgf Plot({line_width = LINE_WIDTH}, Coordinates(yearlytime, Mpath))

    push!(mfig, medianplot)

    @pgf push!(gfig, {
            figsize...,
            grid = "both",
            ylabel = L"\footnotesize CO2e concentration \\$M_t^{\textrm{np}} \; [\si{\ppm}]$",
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

begin # Carbon decay path
    mediandecay = [100 * δₘ(M, decay) for M in Mpath]
    ytick = 0:0.1:1
    yticklabels = ["$(round(δ, digits = 2)) \\%" for δ in ytick]

    decaypathfig = @pgf Axis({
        width = raw"0.595\textwidth",
        height = raw"0.425\textwidth",
        grid = "both",
        xlabel = raw"Carbon concentration $M$",
        ylabel = raw"Decay of CO$_2$ in the atmosphere $\delta_m$",
        xmin = minimum(Mpath), xmax = maximum(Mpath),
        scaled_y_ticks = false,
        ytick = ytick, yticklabels = yticklabels
    })

    @pgf push!(decaypathfig,
        Plot({line_width = LINE_WIDTH}, Coordinates(Mpath, mediandecay))
    )

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "decaypathfig.tikz"), decaypathfig; include_preamble=true)
    end

    decaypathfig
end

let # Damage fig
    activedamages = [d(T, linearmodel.economy.damages) for T in Tspace]
    comparedamages = [
        (raw"Kalkuhl-Wenz (2020) preferred", [d(T, Kalkuhl{Float64}()) for T in Tspace], "solid", "square*"),
        (raw"Weitzman (2012)", [d(T, WeitzmanGrowth{Float64}()) for T in Tspace], "solid", "triangle*"),
        (raw"Dell et al. (2012)", [d(T, QuadraticDamages(0.002131, 0.)) for T in Tspace], "solid", "diamond*")
    ]

    maxpercentage = ceil(max(maximum(activedamages), maximum(maximum(curve) for (_, curve, _, _) in comparedamages)), digits=2)
    ytick = 0:0.02:maxpercentage
    yticklabels = [@sprintf("%.0f \\%%", 100 * y) for y in ytick]

    _, xticklabels = makedeviationtickz(Tspace[1], Tspace[end]; step = 1, digits = 0)
    xtick = Tspace[1]:1:Tspace[end]

    damagefig = @pgf Axis({
        width = raw"0.7\textwidth",
        height = raw"0.425\textwidth",
        grid = "both",
        xlabel = TLABEL,
        ylabel = raw"Damage function $d(T_t) ; [\si{1 / year}]$",
        xmin = 0, xmax = Tspace[end],
        xticklabel_style = {rotate = 45},
        yticklabels = yticklabels, ytick = ytick, ymin = 0.,
        xticklabels = xticklabels, xtick = xtick,
        scaled_y_ticks = false,
        legend_style = {at = {"(0.03,0.97)"}, anchor = "north west", nodes = {scale = 0.75}},
        legend_cell_align = "left",
        ymin = 0, ymax = 0.1,
    })

    for (label, curve, style, marker) in comparedamages
        comparedcurve = @pgf Plot({
                line_width = LINE_WIDTH / 2,
                color = "black",
                opacity = 0.7,
                style = style,
                mark = marker,
                mark_repeat = 10,
                mark_size = 1.9,
                mark_options = {fill = "white", draw = "black"}
            },
            Coordinates(Tspace, curve)
        )
        push!(damagefig, comparedcurve, LegendEntry(label))
    end

        @pgf damagecurve = Plot({line_width = LINE_WIDTH + 0.4, color = "black"},
        Coordinates(Tspace, activedamages)
    )

    push!(damagefig, damagecurve, LegendEntry("This paper"))

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "damagefig.tikz"), damagefig; include_preamble=true)
    end

    damagefig
end

# Growth vs Level
function deterministictemperature(t, climate::LinearClimate, γ₀)
    @unpack S₀, G₀, G₁, Tᵖ, η, σ = climate.hogg

    return ((G₀ + S₀ + G₁ * γ₀ * t) / η)^(1 / 4) - Tᵖ
end

function instantenousdamages(u, p, t)
    linearmodel, growthdamage, γ₀ = p
    Tₜ = deterministictemperature(t, linearmodel.climate, γ₀)

    return d(Tₜ, growthdamage)
end

"Generate level damage comparison axis for a given growth rate γ₀."
function level_damage_axis(γ₀, linearmodel, Tspace; withlegend = false, withylabel = false)
    Tmax = Tspace[end]
    t̄ = find_zero(t -> deterministictemperature(t, linearmodel.climate, γ₀) - Tmax, (0., 5000.))

    cumulativedamageprob = ODEProblem(instantenousdamages, 0., (0., t̄))
    damagetraj = solve(cumulativedamageprob; p = (linearmodel, BurkeHsiangMiguel(), γ₀))
    delltraj = solve(cumulativedamageprob; p = (linearmodel, WeitzmanGrowth{Float64}(), γ₀))

    timespace = range(cumulativedamageprob.tspan..., 101)
    Tₜ = [deterministictemperature(t, linearmodel.climate, γ₀) for t in timespace]
    Dₜ = [1 - exp(-damagetraj(t)) for t in timespace]
    Dₜdell = [1 - exp(-delltraj(t)) for t in timespace]

    comparedamages = [
        ("DICE (Nordhaus, 2017)", [D(T, DICE()) for T in Tₜ], "solid", "square*"),
        ("Weitzman Level (2012)", [1 - D(T, WeitzmanLevel()) for T in Tₜ], "solid", "triangle*"),
        ("Weitzman Growth (2012)", Dₜdell, "solid", "diamond*")
    ]

    ytick = 0:0.2:1.0
    yticklabels = [@sprintf("%.0f\\%%", 100 * y) for y in ytick]

    _, xticklabels = makedeviationtickz(Tₜ[1], Tₜ[end]; step = 1, digits = 0)
    xtick = floor(Tₜ[1]):1:ceil(Tₜ[end])

    axis = @pgf Axis({
        width = raw"0.45\linewidth",
        height = raw"0.36\linewidth",
        grid = "both",
        xlabel = TLABEL,
        ylabel = withylabel ? raw"Level damage $D_t$" : "",
        xmin = Tₜ[1], xmax = Tₜ[end],
        xticklabels = xticklabels, xtick = xtick,
        xticklabel_style = {rotate = 45},
        yticklabels = yticklabels, ytick = ytick,
        ymin = 0., ymax = 1.,
        scaled_y_ticks = false,
        legend_style = {at = {"(0.03, 0.97)"}, anchor = "north west", nodes = {scale = 0.6}},
        legend_cell_align = "left"
    })

    for (label, curve, style, marker) in comparedamages
        comparedcurve = @pgf Plot({
                line_width = LINE_WIDTH / 2,
                color = "black",
                opacity = 0.7,
                style = style,
                mark = marker,
                mark_repeat = 10,
                mark_size = 1.9,
                mark_options = {fill = "white", draw = "black"}
            },
            Coordinates(Tₜ, curve)
        )
        push!(axis, comparedcurve)
        withlegend && push!(axis, LegendEntry(label))
    end

    activecurve = @pgf Plot({line_width = LINE_WIDTH + 0.4, color = "black"},
        Coordinates(Tₜ, Dₜ)
    )
    push!(axis, activecurve)
    withlegend && push!(axis, LegendEntry("This paper"))

    return axis
end

# Level damage comparison across growth scenarios
begin
    γ₀scenarios = [
        (0.014, L"\bar{\gamma} \equiv \gamma_0"),
        (0.028, L"\bar{\gamma} \equiv 2\gamma_0"),
    ]

    groupfig = @pgf GroupPlot({
        group_style = {
            group_size = "2 by 1",
            horizontal_sep = "4em"
        }
    })

    for (k, (γ, label)) in enumerate(γ₀scenarios)
        axis = level_damage_axis(γ, linearmodel, Tspace; withlegend = k == length(γ₀scenarios), withylabel = k == 1)
        push!(groupfig, axis)
        @pgf axis["title"] = label
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "leveldamage-scenarios.tikz"), groupfig; include_preamble=true)
    end

    groupfig
end

begin # Marginal abatement curve
    emissivity = range(0.0, 1.1; length = 51)

    times = [0., 10., 20.] |> reverse

    yearcolors = get(PALETTE, (times) / 30.)

    xticks = range(extrema(emissivity)..., step = 0.2)
    xticklabels = [@sprintf("%.0f\\%%", 100 * x) for x in xticks]

    ytick = 0:0.1:0.5
    yticklabels = [@sprintf("%.f\\%%", 100 * y) for y in ytick]

    abatementfig = @pgf Axis({
        width = raw"0.604\textwidth",
        height = raw"0.425\textwidth",
        grid = "both",
        xlabel = L"Abated percentage $\varepsilon(\alpha_t)$",
        ylabel = L"Abatement costs $\beta_t\big(\varepsilon(\alpha_t)\big)$",
        xmin = 0., xmax = maximum(emissivity),
        xtick = xticks, xticklabels = xticklabels,
        ymin = 0., ymax = maximum(ytick),
        ytick = ytick, yticklabels = yticklabels,
        scaled_y_ticks = false
    })

    # Add gray band for ε > 1 (negative emissions)
    bandx = [1.0, maximum(emissivity)]
    bandy = [0.0, maximum(ytick)]
    bandcoords = vcat([(x, bandy[1]) for x in bandx], [(x, bandy[2]) for x in reverse(bandx)])
    bandpoly = @pgf Plot({fill = "gray", opacity = 0.25, draw = "none", forget_plot}, Coordinates(bandcoords))
    push!(abatementfig, bandpoly)

    for (k, t) in enumerate(times)
        mac = [β(t, ε, abatement) for ε in emissivity]

        abatementcurve = @pgf Plot({line_width = LINE_WIDTH, color = yearcolors[k]}, Coordinates(emissivity, mac))

        push!(abatementfig, abatementcurve, LegendEntry(@sprintf("%d", 2020 + t)))
    end

    @pgf abatementfig["legend style"] = raw"at = {(0.3, 0.95)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "abatementfig.tikz"), abatementfig; include_preamble=true)
    end
    
    abatementfig
end