using PGFPlotsX
using Plots
using Roots
using LaTeXStrings
using DotEnv, JLD2
using FastClosures

using Random, DataStructures

using Plots, PGFPlotsX
using LaTeXStrings, Printf
using Colors, ColorSchemes

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}", raw"\usetikzlibrary{patterns}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{siunitx}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\DeclareSIUnit{\ppm}{p.p.m.}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\DeclareSIUnit{\CO}{\,CO_2}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\DeclareSIUnit{\output}{trillion US\mathdollar / year}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\DeclareSIUnit{\shortoutput}{tr US\mathdollar / y}")

using Statistics
using Model, Grid
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations: Extrapolation
using Dierckx, ImageFiltering

includet("../utils.jl")
includet("../../utils/saving.jl")
includet("../../utils/simulating.jl")

begin # Environment variables
    env = DotEnv.config(".env.game")
    plotpath = get(env, "PLOTPATH", "plots")
    datapath = get(env, "DATAPATH", "data")
    simulationpath = get(env, "SIMULATIONPATH", "simulations")

    BASELINE_YEAR = 2020
    PLOT_HORIZON = 60.
    LINE_WIDTH = 2.5
    SAVEFIG = false

    decadetick = 0:10:Int(PLOT_HORIZON)
    decadeticklabels = decadetick .+ BASELINE_YEAR

    calibration = load_object(joinpath(datapath, "calibration.jld2"))
    regionalcalibration = load_object(joinpath(datapath, "regionalcalibration.jld2"))
end;


begin # Models definition
    # -- Climate
    hogg = Hogg()
    # -- Economy and Preferences
    preferences = EpsteinZin();
    oecdeconomy, roweconomy = RegionalEconomies()
    damages = GrowthDamages()

    oecdmodel = LinearModel(hogg, preferences, damages, oecdeconomy)

    rowmodels = SortedDict{Float64, AbstractModel}()    
    thresholds = [1.8, 2., 2.5]

    for threshold in thresholds
        rowmodels[threshold] = TippingModel(Albedo(threshold), hogg, preferences, damages, roweconomy);
    end

    rowmodels[Inf] = LinearModel(hogg, preferences, damages, roweconomy)
    push!(thresholds, Inf)
end;

begin # Load simulations and build interpolations
    Interpolation = Dict{AbstractModel, Dict{Symbol, Extrapolation}}
    interpolations = SortedDict{Float64, Interpolation}();

    for (threshold, rowmodel) in rowmodels
        result = loadgame(AbstractModel[oecdmodel, rowmodel]; outdir = simulationpath)

        interpolations[threshold] = buildinterpolations(result)
    end
end

begin # Labels, colors and axis
    PALETTE = cgrad(:Reds, rev = true)
    colors = get(PALETTE, range(0, 0.5; length = length(rowmodels)))
    
    oecdcolor = RGB(2 / 255, 57 / 255, 74 / 255)

    colorsbymodels = Dict{AbstractModel, RGB{Float64}}(values(rowmodels) .=> colors)
    colorsbythreshold = SortedDict{Float64, RGB{Float64}}(thresholds .=> colors)

    ΔTmax = 6.
    ΔTspace = range(0.0, ΔTmax; length = 51)
    Tspace = ΔTspace .+ Hogg().Tᵖ

    horizon = round(Int64, calibration.tspan[2])
    yearlytime = 0:1:horizon

    temperatureticks = collect.(makedeviationtickz(0.0, ΔTmax, oecdmodel; step=1, digits=0))

    Tmin, Tmax = extrema(temperatureticks[1])
end;

# -- Make simulation of optimal trajectories
begin
    TRAJECTORIES = 100_000;
    simulations = SortedDict{Float64, EnsembleSolution}();

    # The initial state is given by (T₁, T₂, m, y₁, y₂)
    X₀ = [Hogg().T₀, Hogg().T₀, log(Hogg().M₀), log(oecdeconomy.Y₀), log(roweconomy.Y₀)];

    for (threshold, itp) in interpolations
        rowmodel = rowmodels[threshold]

        oecdpolicies = (itp[oecdmodel][:χ], itp[oecdmodel][:α]);
        rowpolicies = (itp[rowmodel][:χ], itp[rowmodel][:α]);

        policies = (oecdpolicies, rowpolicies);
        models = (oecdmodel, rowmodel);

        parameters = (models, policies, calibration);

        problem = SDEProblem(Fgame!, Ggame!, X₀, (0., PLOT_HORIZON), parameters)

        ensembleprob = EnsembleProblem(problem)

        simulation = solve(ensembleprob; trajectories = TRAJECTORIES);
        println("Done with simulation of $threshold")

        simulations[threshold] = simulation
    end
end

# Results extraction
using DifferentialEquations.EnsembleAnalysis

quantiles = [0.1, 0.3, 0.5, 0.7, 0.9];
medianidx = findfirst(q -> q == 0.5, quantiles);

timesteps = 0:0.5:PLOT_HORIZON;
decades = 0:10:Int(PLOT_HORIZON);

# Extract quantiles
quantilesdict = Dict{Float64, DiffEqArray}();
for (threshold, sim) in simulations
    quantilesdict[threshold] = timeseries_point_quantile(sim, quantiles, timesteps)
end;

# Extract control quantiles
poldict = SortedDict{Float64, Dict{Symbol, Matrix{Float64}}}();
for (threshold, itp) in interpolations
    ε₁ = @closure (T₁, T₂, m, y₁, y₂, t) -> begin
        α = itp[oecdmodel][:α](T₁, m, t)

        return ε(t, exp(m), α, oecdmodel, regionalcalibration, 1)
    end

    ε₂ = @closure (T₁, T₂, m, y₁, y₂, t) -> begin
        rowmodel = rowmodels[threshold]
        α = itp[rowmodel][:α](T₂, m, t)

        return ε(t, exp(m), α, rowmodel, regionalcalibration, 2)
    end

    gap = @closure (T₁, T₂, m, y₁, y₂, t) -> begin
        α₁ = itp[oecdmodel][:α](T₁, m, t)

        rowmodel = rowmodels[threshold]
        α₂ = itp[rowmodel][:α](T₂, m, t)

        return γ(t, regionalcalibration.calibration) - α₁ - α₂
    end
    
    A₁ = computeonsim(simulations[threshold], ε₁, timesteps);
    A₂ = computeonsim(simulations[threshold], ε₂, timesteps);
    G = computeonsim(simulations[threshold], gap, timesteps);

    polquantiles = Dict{Symbol, Matrix{Float64}}()
    polquantiles[:oecd] = Array{Float64}(undef, length(axes(A₁, 1)), length(quantiles))
    polquantiles[:row] = Array{Float64}(undef, length(axes(A₂, 1)), length(quantiles))
    polquantiles[:gap] = Array{Float64}(undef, length(axes(G, 1)), length(quantiles))

    for tdx in axes(A₁, 1)
        polquantiles[:oecd][tdx, :] .= Statistics.quantile(A₁[tdx, :], quantiles)
        polquantiles[:row][tdx, :] .= Statistics.quantile(A₂[tdx, :], quantiles)
        polquantiles[:gap][tdx, :] .= Statistics.quantile(G[tdx, :], quantiles)
    end

    poldict[threshold] = polquantiles
end

begin # Control
    ytick = 0:0.2:1.0
    yticklabels = [@sprintf("%.0f\\%%", 100y) for y in ytick]

    policyfig = @pgf GroupPlot({
        group_style = {
            group_size = "2 by 1",
            yticklabels_at = "edge left",
            horizontal_sep = "10pt"
        }
    })
    
    for region in [:oecd, :row]
        isoecd = region == :oecd

        regionopts = @pgf isoecd ? {
            title = "OECD",
            xlabel = "Year",
            ylabel = L"Fraction of abated emissions $\varepsilon_i(\alpha_{i, t})$",
            ytick = ytick, yticklabels = yticklabels,
            xlabel_style = { anchor = "north", xshift = "105pt" },
            xtick = decadetick[1:(end-1)], xticklabels = decadeticklabels[1:(end-1)]
        } : {
            title = "RoW", ytick = [],
            xtick = decadetick, xticklabels = decadeticklabels
        }

        abatementfig = @pgf Axis({
            grid = "both",
            ymin = 0., ymax = 1.05,
            xmin = 0, xmax = PLOT_HORIZON,
            xticklabel_style = { rotate = 45 },
            legend_style = { at = "{(0.5, 0.95)}", font = raw"\small"},
            regionopts...
        })

        for (threshold, policies) in poldict
            simquantiles = policies[region]
            median = simquantiles[:, medianidx]

            color = colorsbythreshold[threshold]

            forget_plot = @pgf isoecd ? {} : {forget_plot}
            
            mediancoords = Coordinates(timesteps, median)
            medianpath = @pgf Plot({ color = color, forget_plot..., line_width = LINE_WIDTH}, mediancoords)

            if isoecd
                label = isfinite(threshold) ? @sprintf("\$T^c = %.1f \\si{\\degree} \$", threshold) : "No Tipping"

                push!(abatementfig, LegendEntry(label))
            end
            
            markers = @pgf Plot({ only_marks, mark_options = {scale = 1.5, draw_opacity = 0 }, color = color, forget_plot, mark_repeat = 10}, mediancoords)

            push!(abatementfig, medianpath, markers)

            qdx = medianidx
            for dxstep in 1:(length(quantiles) ÷ 2)
                opacity = 0.1 + 0.3 * (dxstep / length(quantiles))

                lowerpath = @pgf Plot(
                    {draw = "none", name_path = "lower", forget_plot}, 
                    Coordinates(timesteps, simquantiles[:, qdx - dxstep]))
            
                upperpath = @pgf Plot(
                    {draw = "none", name_path = "upper", forget_plot}, 
                    Coordinates(timesteps, simquantiles[:, qdx + dxstep]))
            
                shading = @pgf Plot({opacity = opacity, fill = color, forget_plot}, "fill between [of=lower and upper]")

                push!(abatementfig, lowerpath, upperpath, shading)
            end
        end;

        push!(policyfig, abatementfig)
    end
    
    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "optabatement.tikz"), policyfig; include_preamble=true)
    end

    policyfig
end

begin # Net growth of CO2
    ytick = -0.01:0.005:0.01
    yticklabels = [@sprintf("%.1f\\%%", 100y) for y in ytick]

    growthfig = @pgf Axis({
        grid = "both",
        xmin = 0, xmax = PLOT_HORIZON,
        xtick = decadetick, xticklabels = decadeticklabels,
        xticklabel_style = { rotate = 45 },
        ylabel = L"Growth rate of CO$_2$ concentration",
        ytick = ytick, yticklabels = yticklabels,
        legend_style = { font = raw"\small"},
        scaled_y_ticks = false
    })

    zeroline = @pgf HLine({ color = "black", line_width = LINE_WIDTH }, 0.)
    push!(growthfig, zeroline)

    for (threshold, policies) in poldict
        simquantiles = policies[:gap]
        median = simquantiles[:, medianidx]

        color = colorsbythreshold[threshold]

        mediancoords = Coordinates(timesteps, median)
        medianpath = @pgf Plot({ color = color, line_width = LINE_WIDTH }, mediancoords)

        label = isfinite(threshold) ? @sprintf("\$T^c = %.1f \\si{\\degree} \$", threshold) : "No Tipping"

        push!(growthfig, medianpath, LegendEntry(label))
        
        markers = @pgf Plot({ only_marks, mark_options = {scale = 1.5, draw_opacity = 0 }, color = color, forget_plot, mark_repeat = 10 }, mediancoords)

        push!(growthfig, markers)

        qdx = medianidx
        for dxstep in 1:(length(quantiles) ÷ 2)
            opacity = 0.1 + 0.3 * (dxstep / length(quantiles))

            lowerpath = @pgf Plot({ draw = "none", name_path = "lower", forget_plot }, Coordinates(timesteps, simquantiles[:, qdx - dxstep]))
        
            upperpath = @pgf Plot({ draw = "none", name_path = "upper", forget_plot }, Coordinates(timesteps, simquantiles[:, qdx + dxstep]))
        
            shading = @pgf Plot({ opacity = opacity, fill = color, forget_plot }, "fill between [of=lower and upper]")

            push!(growthfig, lowerpath, upperpath, shading)
        end
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "growth.tikz"), growthfig; include_preamble=true)
    end

    growthfig
end

begin # State
    statesfig = @pgf GroupPlot({
        group_style = {
            group_size = "2 by 2",
            yticklabels_at = "edge left",
            horizontal_sep = "10pt",
            vertical_sep = "10pt"
        }
    })
    
    for statedx in [1, 4], region in [:oecd, :row]
        isoecd = region == :oecd
        if !isoecd statedx += 1 end
        istemperature = statedx < 3

        isfirst = istemperature && isoecd
        issecond = istemperature && !isoecd
        isthird = !istemperature && isoecd
        islast = !istemperature && !isoecd

        stateopts = @pgf istemperature ? {
            ymin = Tmin, ymax = Tmax, xtick = []
        } : {
            xtick = decadetick, xticklabels = decadeticklabels,
            ymin = roweconomy.Y₀, ymax = 4roweconomy.Y₀
        }

        firstopts = @pgf isfirst ? {
            ylabel = L"Temperature $T_{i, t} \; [\si{\degree}]$",
            ytick = temperatureticks[1], yticklabels = temperatureticks[2], title = "OECD"
        } : {}

        secondopts = @pgf issecond ? {
            title = "RoW"
        } : {}

        thirdopts = @pgf isthird ? {
            ylabel = L"Output $Y_{i, t} \; [\si{\output}]$"
        } : {}

        regionopts = @pgf islast ? {
            xlabel = "Year",
            xlabel_style = { anchor = "north", xshift = "105pt" }
        } : {}

        statefig = @pgf Axis({
            grid = "both",
            xmin = 0, xmax = PLOT_HORIZON,
            xticklabel_style = { rotate = 45 },
            legend_style = { at = "{(0.5, 0.95)}", font = raw"\small" },
            firstopts..., secondopts..., thirdopts..., 
            stateopts..., regionopts...,
        })

        for (threshold, sim) in quantilesdict
            statequantiles = getindex.(sim.u, statedx)

            median = getindex.(statequantiles, medianidx)

            if !istemperature median = exp.(median) end

            color = colorsbythreshold[threshold]

            forget_plot = @pgf isfirst ? {} : { forget_plot }
            
            mediancoords = Coordinates(timesteps, median)
            medianpath = @pgf Plot({ color = color, forget_plot..., line_width = LINE_WIDTH}, mediancoords)

            if isfirst
                label = isfinite(threshold) ? @sprintf("\$T^c = %.1f \\si{\\degree} \$", threshold) : "No Tipping"

                push!(statefig, LegendEntry(label))
            end
            
            markers = @pgf Plot({ only_marks, mark_options = {scale = 1.5, draw_opacity = 0 }, color = color, forget_plot, mark_repeat = 10}, mediancoords)

            push!(statefig, medianpath, markers)

            qdx = medianidx
            for dxstep in 1:(length(quantiles) ÷ 2)
                opacity = 0.1 + 0.3 * (dxstep / length(quantiles))

                lower = getindex.(statequantiles, qdx - dxstep)
                upper = getindex.(statequantiles, qdx + dxstep)

                if !istemperature
                    lower = exp.(lower)
                    upper = exp.(upper)
                end

                lowerpath = @pgf Plot(
                    {draw = "none", name_path = "lower", forget_plot}, 
                    Coordinates(timesteps, lower))
            
                upperpath = @pgf Plot(
                    {draw = "none", name_path = "upper", forget_plot}, 
                    Coordinates(timesteps, upper))
            
                shading = @pgf Plot({opacity = opacity, fill = color, forget_plot}, "fill between [of=lower and upper]")

                push!(statefig, lowerpath, upperpath, shading)
            end
        end;

        push!(statesfig, statefig)
    end
    
    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "statefig.tikz"), policyfig; include_preamble=true)
    end

    statesfig
end

