using Revise
using JLD2, UnPack, DataStructures
using FastClosures
using Base.Threads
using SciMLBase
using Statistics
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations, ForwardDiff
using StaticArrays

using Model, Grid
using Random; Random.seed!(11148705);

using Plots, PGFPlotsX, Contour
using LaTeXStrings, Printf
using Colors, ColorSchemes

# pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE,
    raw"\usepgfplotslibrary{fillbetween}",
    raw"\usetikzlibrary{patterns}",
    raw"\usepackage{siunitx}",
    raw"\DeclareSIUnit{\ppm}{p.p.m.}",
    raw"\DeclareSIUnit{\CO}{\,CO_2e}",
    raw"\DeclareSIUnit{\output}{trillion US\mathdollar / year}",
    raw"\DeclareSIUnit{\shortoutput}{tr US\mathdollar / y}",
)

includet("../../../src/valuefunction.jl")
includet("../../../src/extend/model.jl")
includet("../../../src/extend/grid.jl")
includet("../../../src/extend/valuefunction.jl")
includet("../utils.jl")
includet("../../utils/saving.jl")
includet("../../utils/simulating.jl")

CEPATH = "data/ce/simulation-dense"; @assert isdir(CEPATH)
DATAPATH = "data/simulation-dense"; @assert isdir(DATAPATH)

SAVEFIG = true;
PLOTPATH = "../job-market-paper/submission/plots"
plotpath = joinpath(PLOTPATH, "negative")
if !isdir(plotpath) mkpath(plotpath) end

linearmodel, G = joinpath(DATAPATH, "linear/growth/logseparable/negative/Linear_burke_RRA10,00.jld2") |> loadproblem


begin # Plot estetics
    PALETTE = colorschemes[:grays]

    TEMPLABEL = raw"Temperature deviations $T_t \; [\si{\degree\Celsius}]$"
    LINE_WIDTH = 2.5

    Tspace, mspace = G.ranges

    T₀ = linearmodel.climate.hogg.T₀
    m₀ = log(linearmodel.climate.hogg.M₀ / linearmodel.climate.hogg.Mᵖ)
    x₀ = Point(T₀, m₀)

    temperatureticks = makedeviationtickz(Tspace[1], Tspace[end]; step=1, digits=2)

    X₀ = SVector(T₀, m₀, 0.)
    u₀ = SVector(X₀..., 0., 0., 0.) # Introduce three 0s for costs
end;


begin # SCC
    thresholds = Float64[]
    sccs = Float64[]
    scclinear = NaN

    for model in models
        Hitp, _ = interpolations[model]
        m₀ = log(model.climate.hogg.M₀ / model.climate.hogg.Mᵖ)
        ∂Hₘ = ForwardDiff.derivative(m -> Hitp(model.climate.hogg.T₀, m, 0.), m₀)
        s = scc(∂Hₘ, model.economy.Y₀, model.climate.hogg.M₀, model)
        
        if model.climate isa TippingClimate
            push!(sccs, s)
            push!(thresholds, model.climate.feedback.Tᶜ)
        else
            scclinear = s
        end
    end
end

begin # SCC surfaces combined
    figdiscoveries = discoveries
    figthresholds = thresholds
    
    sccmatrix = fill(NaN, length(figthresholds), length(figdiscoveries))
    
    for (i, threshold) in enumerate(figthresholds)
        for (j, discovery) in enumerate(figdiscoveries)
            sccmatrix[i, j] = sccs[i] * discoverygradient[i, j][2] / truegradient[i][2]
        end
    end
    
    discoveryticks, discoverylabels = makedeviationtickz(minimum(figdiscoveries), maximum(figdiscoveries); step=0.5, digits=1)
    thresholdticks, thresholdlabels = makedeviationtickz(minimum(figthresholds), maximum(figthresholds); step=0.5, digits=1)
    
    # Remove last tick from y-axis
    thresholdticks = thresholdticks[1:end-1]
    thresholdlabels = thresholdlabels[1:end-1]
    
    sccsurfacefig = @pgf Axis({
        xlabel = L"Discovery temperature $\Delta T^{\mathrm{d}}$ [\si{\degree}]",
        ylabel = L"Critical threshold $T^c$ [\si{\degree}]",
        zlabel = L"SCC $[\si{US\mathdollar / tCe}]$",
        xlabel_style = "{sloped}",
        ylabel_style = "{sloped}",
        zlabel_style = "{sloped}",
        view = "{60}{50}",
        grid = "both",
        xmin = minimum(figdiscoveries),
        xmax = maximum(figdiscoveries),
        ymin = minimum(figthresholds),
        ymax = maximum(figthresholds),
        xtick = discoveryticks,
        xticklabels = discoverylabels,
        ytick = thresholdticks,
        yticklabels = thresholdlabels,
        y_dir = "reverse", x_dir = "reverse",
        width = "0.7225\\textwidth",
        height = "0.7225\\textwidth",
        ztick_distance = 10,
        legend_pos = "north east"
    })
    
    ploteverythreshold = max(1, length(figthresholds) ÷ 6)
    ploteverydiscovery = max(1, length(figdiscoveries) ÷ 6)
    
    # Filled baseline surface
    baselinesurface = @pgf Plot3({
        surf,
        opacity = 0.3,
        color = "gray",
        shader = "flat",
        forget_plot
    }, Table(figdiscoveries, figthresholds, sccs))
    
    push!(sccsurfacefig, baselinesurface)
    
    # Lines along discovery direction (constant threshold)
    firstdiscoveryline = true
    firstbaselineline = true
    
    for (i, threshold) in enumerate(figthresholds)
        if (i - 1) % ploteverythreshold != 0 && i != length(figthresholds)
            continue
        end
        
        sccline = sccmatrix[i, :]
        
        lineplot = @pgf Plot3({
            no_marks,
            color = "black",
            line_width = "1.5pt",
            forget_plot = !firstdiscoveryline
        }, Table(x = figdiscoveries, y = fill(threshold, length(figdiscoveries)), z = sccline))
        
        push!(sccsurfacefig, lineplot)
        if firstdiscoveryline
            push!(sccsurfacefig, LegendEntry("Discovery"))
            firstdiscoveryline = false
        end
        
        baseline = @pgf Plot3({
            no_marks,
            dotted,
            color = "gray",
            line_width = "2pt",
            forget_plot = !firstbaselineline
        }, Table(x = figdiscoveries, y = fill(threshold, length(figdiscoveries)), z = fill(sccs[i], length(figdiscoveries))))
        
        push!(sccsurfacefig, baseline)
        if firstbaselineline
            push!(sccsurfacefig, LegendEntry("Optimal"))
            firstbaselineline = false
        end
    end
    
    # Lines along threshold direction (constant discovery)
    for (j, discovery) in enumerate(figdiscoveries)
        if (j - 1) % ploteverydiscovery != 0 && j != length(figdiscoveries)
            continue
        end
        
        sccline = sccmatrix[:, j]
        
        lineplot = @pgf Plot3({
            no_marks,
            color = "black",
            line_width = "1.5pt",
            forget_plot
        }, Table(x = fill(discovery, length(figthresholds)), y = figthresholds, z = sccline))
        
        push!(sccsurfacefig, lineplot)
        
        baseline = @pgf Plot3({
            no_marks,
            dotted,
            color = "gray",
            line_width = "2pt",
            forget_plot
        }, Table(x = fill(discovery, length(figthresholds)), y = figthresholds, z = sccs))
        
        push!(sccsurfacefig, baseline)
    end
    
    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "scc_surface.tikz"), sccsurfacefig; include_preamble=true)
    end
    
    sccsurfacefig
end

begin # SCC percent difference surface
    # relies on sccmatrix, sccs, figdiscoveries, figthresholds already created
    percentdiffmatrix = fill(NaN, size(sccmatrix))
    for i in eachindex(figthresholds)
        baselinevalue = sccs[i]
        for j in eachindex(figdiscoveries)
            val = sccmatrix[i, j]
            if !isnan(val) && !isnan(baselinevalue) && baselinevalue != 0.0
                percentdiffmatrix[i, j] = 100.0 * (val - baselinevalue) / baselinevalue
            end
        end
    end

    discoveryticks, discoverylabels = makedeviationtickz(minimum(figdiscoveries), maximum(figdiscoveries); step=0.5, digits=1)
    thresholdticks, thresholdlabels = makedeviationtickz(minimum(figthresholds), maximum(figthresholds); step=0.5, digits=1)
    thresholdticks = thresholdticks[1:end-1]
    thresholdlabels = thresholdlabels[1:end-1]

    ztick = 0:10:50
    zticklabels = [ @sprintf("\\footnotesize %.0f\\%%", z) for z in ztick ]

    percentdiffaxis = @pgf Axis({
        xlabel = L"Discovery temperature $\Delta T^{\mathrm{d}}$ [\si{\degree}]",
        ylabel = L"Critical threshold $T^c$ [\si{\degree}]",
        xlabel_style = "{sloped}",
        ylabel_style = "{sloped}",
        view = "{60}{50}",
        grid = "both",
        xmin = minimum(figdiscoveries), xmax = maximum(figdiscoveries),
        ymin = minimum(figthresholds), ymax = maximum(figthresholds),
        zmin = 0, zmax = 50,
        xtick = discoveryticks,
        xticklabels = discoverylabels,
        ztick = ztick, zticklabels = zticklabels,
        ytick = thresholdticks,
        yticklabels = thresholdlabels,
        x_dir = "reverse", y_dir  = "reverse",
        width = "0.7225\\textwidth",
        height = "0.7225\\textwidth"
    })

    # Subsample the data for coarser mesh
    meshstep = 1
    meshdiscoveries = figdiscoveries[1:meshstep:end]
    meshthresholds = figthresholds[1:meshstep:end]
    meshpercentdiff = percentdiffmatrix[1:meshstep:end, 1:meshstep:end]
    
    diffsurface = @pgf Plot3({
        mesh,
        color = "black",
        forget_plot
    }, Table(meshdiscoveries, meshthresholds, meshpercentdiff'))

    push!(percentdiffaxis, diffsurface)

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "scc_surface_percentdiff.tikz"), percentdiffaxis; include_preamble=true)
    end

    percentdiffaxis
end