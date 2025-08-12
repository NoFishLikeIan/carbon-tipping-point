using Revise

using DotEnv, JLD2
using Random, DataStructures
using Base.Order: Reverse

using PGFPlotsX, Plots
using LaTeXStrings, Printf
using Colors, ColorSchemes

push!(PGFPlotsX.CUSTOM_PREAMBLE, 
    raw"\usepgfplotslibrary{fillbetween}",
    raw"\usetikzlibrary{patterns}",
    raw"\usepackage{siunitx}",
    raw"\usepackage{amsmath}"
)

push!(PGFPlotsX.CUSTOM_PREAMBLE,
    raw"\usepackage{siunitx}",
    raw"\DeclareSIUnit{\ppm}{p.p.m.}",
    raw"\DeclareSIUnit{\CO}{\,CO_2}",
    raw"\DeclareSIUnit{\output}{trillion US\mathdollar / year}",
    raw"\DeclareSIUnit{\shortoutput}{tr US\mathdollar / y}",
    raw"\DeclareMathOperator{\oecd}{OECD}",
    raw"\DeclareMathOperator{\row}{RoW}"
)

using Statistics
using Model, Grid
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations: Extrapolation
using Dierckx, FastClosures
using LinearAlgebra, SparseArrays

includet("../utils.jl")
includet("../../utils/saving.jl")
includet("../../utils/simulating.jl")

begin # Environment variables
    env = DotEnv.config(".env.game")
    plotpath = get(env, "PLOTPATH", "plots")
    datapath = get(env, "DATAPATH", "data")
    simulationpath = get(env, "SIMULATIONPATH", "simulations")

    BASELINE_YEAR = 2020
    PLOT_HORIZON = 80.
    LINE_WIDTH = 2.5
    TRAJECTORIES = 10_000;
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
    damages = Kalkuhl()

    oecdmodel = LinearModel(hogg, preferences, damages, oecdeconomy)

    rowmodels = SortedDict{Float64, AbstractModel}(Reverse)
    thresholds = [1.8, 2., 2.5]

    for threshold in thresholds
        rowmodels[threshold] = TippingModel(Albedo(threshold), hogg, preferences, damages, roweconomy);
    end

    rowmodels[Inf] = LinearModel(hogg, preferences, damages, roweconomy)
    push!(thresholds, Inf)
end;

begin # Load simulations and build interpolations
    Interpolation = Dict{AbstractModel, Dict{Symbol, Extrapolation}}
    interpolations = SortedDict{Float64, Interpolation}(Reverse);

    for (threshold, rowmodel) in rowmodels
        models = AbstractModel[oecdmodel, rowmodel]
        result = loadgame(models; outdir = simulationpath)

        interpolations[threshold] = buildinterpolations(result)
    end
end;

begin # Labels, colors and axis
    PALETTE = cgrad(:Reds, rev = true)
    colors = get(PALETTE, range(0, 0.5; length = length(rowmodels)))
    
    oecdcolor = RGB(2 / 255, 57 / 255, 74 / 255)

    colorsbymodels = Dict{AbstractModel, RGB{Float64}}(values(rowmodels) .=> colors)
    colorsbythreshold = SortedDict{Float64, RGB{Float64}}(Reverse, thresholds .=> colors)

    ΔTmax = 6.
    ΔTspace = range(0.0, ΔTmax; length = 51)
    Tspace = ΔTspace .+ Hogg().Tᵖ

    horizon = PLOT_HORIZON
    yearlytime = 0:1:horizon

    temperatureticks = collect.(makedeviationtickz(0.0, ΔTmax, oecdmodel; step=1, digits=0))

    Tmin, Tmax = extrema(temperatureticks[1])
end;

# -- Make simulation of optimal trajectories
begin
    simulations = SortedDict{Float64, EnsembleSolution}();

    # The initial state is given by (T₁, T₂, m, y₁, y₂)
    X₀ = [Hogg().T₀, Hogg().T₀, log(Hogg().M₀), log(oecdeconomy.Y₀), log(roweconomy.Y₀)];

    for (threshold, itp) in interpolations
        rowmodel = rowmodels[threshold]

        oecditps = itp[oecdmodel]
        oecdpolicies = (oecditps[:χ], oecditps[:α]);

        rowitps = itp[rowmodel]
        rowpolicies = (rowitps[:χ], rowitps[:α]);

        policies = (oecdpolicies, rowpolicies);
        models = (oecdmodel, rowmodel);

        parameters = (models, policies, calibration);

        problem = SDEProblem(Fgame!, Ggame!, X₀, (0., 2horizon), parameters)

        ensembleprob = EnsembleProblem(problem)

        simulation = solve(ensembleprob; trajectories = TRAJECTORIES);
        simulations[threshold] = simulation
        
        println("Done with simulation of threshold $(threshold)°")
    end
end;

# Results extraction
using DifferentialEquations.EnsembleAnalysis

quantiles = [0.1, 0.5, 0.9];
medianidx = findfirst(q -> q == 0.5, quantiles);

timesteps = 0:0.5:horizon;
decades = 0:10:Int(horizon);

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
            xlabel_style = { anchor = "north", xshift = raw"0.18\textwidth" },
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
            legend_style = { at = "{(0.7, 0.95)}", font = raw"\footnotesize"},
            width = raw"0.5\textwidth",
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
                opacity = 0.1 + 0.2 * (dxstep / length(quantiles))

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
        legend_style = { at = "{(0.3, 0.35)}", font = raw"\footnotesize"},
        scaled_y_ticks = false,
        width = raw"0.9\textwidth", height = raw"0.5\textwidth"
    })

    zeroline = @pgf HLine({ color = "black", line_width = LINE_WIDTH }, 0.)
    push!(growthfig, zeroline)

    nopolicytraj = Coordinates(timesteps, γ.(timesteps, regionalcalibration.calibration))
    nopolicy = @pgf Plot({color = "black", line_width = LINE_WIDTH, dashed}, nopolicytraj)

    push!(growthfig, nopolicy)

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

begin # Temperature figure
    dynamicsfigure = @pgf GroupPlot({
        group_style = {
            group_size = "2 by 1",
            yticklabels_at = "edge left",
            horizontal_sep = "10pt",
            vertical_sep = "10pt"
        }
    })
    
    for region in [:oecd, :row]
        isoecd = region == :oecd

        regionopts = @pgf isoecd ? {
            ylabel = L"Temperature $T_{i, t} \; [\si{\degree}]$",
            ytick = temperatureticks[1], yticklabels = temperatureticks[2], title = "OECD",
            xtick = decadetick[1:(end-1)], xticklabels = decadeticklabels[1:(end-1)]
        } : { 
            title = "RoW",
            xtick = decadetick, xticklabels = decadeticklabels
        }

        statedx = isoecd ? 1 : 2

        statefig = @pgf Axis({
            grid = "both",
            xmin = 0, xmax = PLOT_HORIZON,
            xticklabel_style = { rotate = 45 },
            legend_style = { at = "{(0.5, 0.95)}", font = raw"\footnotesize" },
            ymin = Tmin, ymax = Tmax, xtick = decadetick, xticklabels = decadeticklabels,
            width = raw"0.5\textwidth", 
            regionopts...
        })

        for (threshold, sim) in quantilesdict
            statequantiles = getindex.(sim.u, statedx)

            median = getindex.(statequantiles, medianidx)

            color = colorsbythreshold[threshold]

            forget_plot = @pgf isoecd ? {} : { forget_plot }
            
            mediancoords = Coordinates(timesteps, median)
            medianpath = @pgf Plot({ color = color, forget_plot..., line_width = LINE_WIDTH}, mediancoords)

            if isoecd
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

        push!(dynamicsfigure, statefig)
    end
    
    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "temperature.tikz"), dynamicsfigure; include_preamble=true)
    end

    dynamicsfigure
end

begin # Cost to convergence
    rowoutput = @pgf Axis({
        grid = "both",
        xmin = 0, xmax = PLOT_HORIZON,
        xtick = decadetick, xticklabels = decadeticklabels,
        xticklabel_style = { rotate = 45 },
        legend_style = { at = "{(0.5, 0.95)}", font = raw"\footnotesize" },
        ylabel = L"Output $Y_{\row, t} \; [\si{\shortoutput}]$",
        width = raw"0.9\textwidth"
    })

    for (threshold, sim) in quantilesdict
        statequantiles = getindex.(sim.u, 5)

        median = exp.(getindex.(statequantiles, medianidx))

        color = colorsbythreshold[threshold]
        
        mediancoords = Coordinates(timesteps, median)
        medianpath = @pgf Plot({ color = color, line_width = LINE_WIDTH}, mediancoords)

        label = isfinite(threshold) ? @sprintf("\$T^c = %.1f \\si{\\degree} \$", threshold) : "No Tipping"

        push!(rowoutput, LegendEntry(label))
        
        markers = @pgf Plot({ only_marks, mark_options = {scale = 1.5, draw_opacity = 0 }, color = color, forget_plot, mark_repeat = 10}, mediancoords)

        push!(rowoutput, medianpath, markers)

        qdx = medianidx
        for dxstep in 1:(length(quantiles) ÷ 2)
            opacity = 0.1 + 0.3 * (dxstep / length(quantiles))

            lower = exp.(getindex.(statequantiles, qdx - dxstep))
            upper = exp.(getindex.(statequantiles, qdx + dxstep))

            lowerpath = @pgf Plot(
                {draw = "none", name_path = "lower", forget_plot}, 
                Coordinates(timesteps, lower))
        
            upperpath = @pgf Plot(
                {draw = "none", name_path = "upper", forget_plot}, 
                Coordinates(timesteps, upper))
        
            shading = @pgf Plot({opacity = opacity, fill = color, forget_plot}, "fill between [of=lower and upper]")

            push!(rowoutput, lowerpath, upperpath, shading)
        end
    end;
    
    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "rowoutput.tikz"), rowoutput; include_preamble=true)
    end

    rowoutput
end

# --- Cost attribution
# -- Favourable OECD scenario
tippingthresholds = thresholds[isfinite.(thresholds)];
function Fjoint!(du, u, parameters::NTuple{2, GameParameters}, t)
    comparam, favparam = parameters
    d = length(u) ÷ 2

    dX₁ = @view du[1:d]
    dX₂ = @view du[(d + 1):2d]

    X₁ = @view u[1:d]
    X₂ = @view u[(d + 1):2d]

    Fgame!(dX₁, X₁, comparam, t)
    Fgame!(dX₂, X₂, favparam, t)

    return
end

function Gjoint!(Σ, u, parameters, t)
    comparam, favparam = parameters
    d = length(u) ÷ 2

    Σ₁ = @view Σ[1:d, 1:d]
    Σ₂ = @view Σ[(d+1):2d, 1:d]
    
    diagΣ₁ = @view Σ₁[diagind(Σ₁)]
    diagΣ₂ = @view Σ₂[diagind(Σ₂)]

    Ggame!(diagΣ₁, u, comparam, t)
    Ggame!(diagΣ₂, u, favparam, t)

    return
end

u₀ = repeat(X₀, 2);
d = length(X₀);
noiseprototype = Matrix(1.0LinearAlgebra.I, d, d);
jointnoiseprototype = spzeros(2d, 2d);
jointnoiseprototype[1:d, 1:d] .= noiseprototype;
jointnoiseprototype[(d+1):2d, 1:d] .= noiseprototype;

begin # Simulation of favourable OECD scenario
    diffsimulations = SortedDict{Float64, EnsembleSolution}();

    oecdfavitp = interpolations[Inf][oecdmodel];
    oecdfavpolicies = (oecdfavitp[:χ], oecdfavitp[:α]);

    for threshold in tippingthresholds
        rowmodel = rowmodels[threshold]

        rowitps = interpolations[threshold][rowmodel]
        rowpolicies = (rowitps[:χ], rowitps[:α]);

        models = (oecdmodel, rowmodel);

        # Define competitive policies
        oecditp = interpolations[threshold][oecdmodel]
        oecdpolicies = (oecditp[:χ], oecditp[:α]);
        compolicies = (oecdpolicies, rowpolicies);
        
        comparam = (models, compolicies, calibration);

        # Define favourable policies
        favpolicies = (oecdfavpolicies, rowpolicies);
        favparams = (models, favpolicies, calibration);

        params = (comparam, favparams);
        problem = SDEProblem(Fjoint!, Gjoint!, u₀, (0., 1.2horizon), params; noise_rate_prototype = jointnoiseprototype);

        ensembleprob = EnsembleProblem(problem);

        diffsimulations[threshold] = solve(ensembleprob; trajectories = TRAJECTORIES)
        
        println("Done with simulation of favourable OECD scenario and threshold $(threshold)°")
    end
end;

begin # Extract output gap in favourable OECD scenario
    outputgap = SortedDict{Float64, Matrix{Float64}}();

    for threshold in tippingthresholds
        gapfn = @closure (T₁, T₂, m, y₁, y₂, Tᶠ₁, Tᶠ₂, mᶠ, yᶠ₁, yᶠ₂, t) -> exp(yᶠ₂) - exp(y₂)

        simulation = diffsimulations[threshold]
            
        gappersim = computeonsim(simulation, gapfn, timesteps);

        outputgap[threshold] = Matrix{Float64}(undef, length(axes(gappersim, 1)), length(quantiles))

        for tdx in axes(gappersim, 1)
            outputgap[threshold][tdx, :] .= Statistics.quantile(gappersim[tdx, :], quantiles)
        end
    end
end;


begin # Output gap figure
    # ytick = 1 .+ (-1e-3:5e-4:4e-3)
    # yticklabels = [@sprintf("%.2f\\%%", 100(y - 1)) for y in ytick]

    outputgapfigure = @pgf Axis({
        grid = "both",
        xmin = 0, xmax = PLOT_HORIZON,
        xtick = decadetick, xticklabels = decadeticklabels,
        xticklabel_style = { rotate = 45 },
        ylabel = L"Output gap $Y^f_{\row, t} - Y_{\row, t} \; \si{\shortoutput}$",
        legend_style = { font = raw"\footnotesize", at = "{(0.3, 0.9)}"},
        scaled_y_ticks = false,
        width = raw"0.9\textwidth", height = raw"0.5\textwidth",
        # ytick = ytick, yticklabels = yticklabels,
        # ymin = minimum(ytick), ymax = maximum(ytick)
    })

    for (threshold, outputgapqs) in outputgap

        median = outputgapqs[:, medianidx]

        color = colorsbythreshold[threshold]

        mediancoords = Coordinates(timesteps, median)
        medianpath = @pgf Plot({ color = color, line_width = LINE_WIDTH }, mediancoords)

        label = isfinite(threshold) ? @sprintf("\$T^c = %.1f \\si{\\degree} \$", threshold) : "No Tipping"

        push!(outputgapfigure, medianpath, LegendEntry(label))
        
        markers = @pgf Plot({ only_marks, mark_options = {scale = 1.5, draw_opacity = 0 }, color = color, forget_plot, mark_repeat = 10 }, mediancoords)

        push!(outputgapfigure, markers)

        if true
            qdx = medianidx
            for dxstep in 1:(length(quantiles) ÷ 2)
                opacity = 0.1 + 0.3 * (dxstep / length(quantiles))

                lowerpath = @pgf Plot({ draw = "none", name_path = "lower", forget_plot }, Coordinates(timesteps, outputgapqs[:, qdx - dxstep]))
            
                upperpath = @pgf Plot({ draw = "none", name_path = "upper", forget_plot }, Coordinates(timesteps, outputgapqs[:, qdx + dxstep]))
            
                shading = @pgf Plot({ opacity = opacity, fill = color, forget_plot }, "fill between [of=lower and upper]")

                push!(outputgapfigure, lowerpath, upperpath, shading)
            end
        end
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "outputgap.tikz"), outputgapfigure; include_preamble=true)
    end

    outputgapfigure
end