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

function findthreshold(threshold, simulationfiles)
    for filepath in simulationfiles
        model = loadproblem(filepath) |> first
        abatementdir = splitpath(filepath)[end - 1]

        istype = (model.economy.damages isa BurkeHsiangMiguel) && (abatementdir == "negative")
        isthreshold = model.climate isa TippingClimate && model.climate.feedback.Tᶜ ≈ threshold


        if istype && isthreshold
            return filepath
        end
    end

    return nothing
end

begin # Import available simulation and CE files
    horizon = 100.
    simulationfiles = listfiles(DATAPATH)

    thresholds = 2:0.05:4;
    discoveries = -1:0.05:1

    truevalue = fill(NaN, length(thresholds))
    truegradient = fill(Point(NaN, NaN), size(truevalue))

    discoveryvalue = fill(NaN, length(thresholds), length(discoveries))
    discoverygradient = fill(Point(NaN, NaN), size(discoveryvalue))

    models = IAM[]
    interpolations = Dict{IAM, NTuple{2, Interpolations.Extrapolation}}()

    missingpairs = Dict{String, Float64}[];

    for (i, threshold) in enumerate(thresholds)
        @printf("Loading threshold=%.2f\r", threshold)

        filepath = findthreshold(threshold, simulationfiles)
        optimalvalues, model, G = loadtotal(filepath; tspan = (0., 1.2horizon))
        Hopt, αopt = buildinterpolations(optimalvalues, G);
        H₀ = Hopt(x₀.T, x₀.m, 0.)
        truevalue[i] = Hopt(x₀.T, x₀.m, 0.)
        truegradient[i] = ForwardDiff.gradient(x ->  Hopt(x[1], x[2], 0.), x₀)

        push!(models, model)
        interpolations[model] = (Hopt, αopt);

        for (j, discovery) in enumerate(discoveries)

            thresholdkey = replace("T$(Printf.format(Printf.Format("%.2f"), threshold))", "." => ",")
            discoverykey = replace("D$(Printf.format(Printf.Format("%.2f"), discovery))", "." => ",")
            outfile = joinpath(CEPATH, "$(thresholdkey)_$(discoverykey).jld2")

            if !isfile(outfile)
                @warn "Outfile $outfile not found!"

                push!(missingpairs, Dict("threshold" => threshold, "discovery" => discovery))

                continue
            end

            JLD2.@load outfile H₀ ∇H₀
            discoveryvalue[i, j] = H₀
            discoverygradient[i, j] = ∇H₀
        end
    end

    linearfilepath = joinpath(DATAPATH, "linear/growth/logseparable/negative/Linear_burke_RRA10,00.jld2")
    linearvalues, linearmodel, linearG = loadtotal(linearfilepath; tspan = (0., 1.2horizon))

    push!(models, linearmodel)
    interpolations[linearmodel] = buildinterpolations(linearvalues, linearG)
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

begin
    Tticks, Tlabels = makedeviationtickz(2, 4; step=0.5, digits=1)
    sccfig = @pgf Axis({ xlabel = L"Critical threshold $T^c$ [\si{\degree}]", ylabel = L"Social cost of carbon in 2020 $[\si{US\mathdollar / tCe}]$", grid = "both", xmin = minimum(thresholds), xmax = maximum(thresholds), xtick = Tticks, xticklabels = Tlabels })

    curve = @pgf Plot({ line_width = LINE_WIDTH }, Coordinates(thresholds, sccs))
    push!(sccfig, curve)

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "scc.tikz"), sccfig; include_preamble=true)
    end

    sccfig
end


begin # SCC by discovery
    figdiscoveries = discoveries
    discoveryticks, discoverylabels = makedeviationtickz(minimum(figdiscoveries), maximum(figdiscoveries); step=0.5, digits=1)
    
    sccbydiscoveryfig = @pgf GroupPlot({
        group_style = {
            group_size = "2 by 2",
            horizontal_sep = "1.5cm",
            vertical_sep = "1.5cm"
        },
        width = "6cm",
        height = "5cm",
        grid = "both",
        xmin = minimum(figdiscoveries),
        xmax = maximum(figdiscoveries),
        xtick = discoveryticks,
        xticklabels = discoverylabels
    })
    
    for (plotidx, threshold) in enumerate(figthresholds)
        idx = searchsortedfirst(thresholds, threshold)
        
        sccbydiscovery = Float64[]
        for (j, discovery) in enumerate(figdiscoveries)
            if !isnan(discoverygradient[idx, j][2]) && !isnan(truegradient[idx][2])
                sccval = sccs[idx] * discoverygradient[idx, j][2] / truegradient[idx][2]
                push!(sccbydiscovery, sccval)
            else
                push!(sccbydiscovery, NaN)
            end
        end
        
        thresholdlabel = @sprintf("%.1f", threshold)
        
        row = (plotidx - 1) ÷ 2 + 1
        col = (plotidx - 1) % 2 + 1
        
        if row == 2 && col == 1
            @pgf push!(sccbydiscoveryfig, {
                title = L"$T^c = %$thresholdlabel$",
                xlabel = L"Discovery $\Delta T^{\mathrm{d}}$ [\si{\degree}]",
                ylabel = L"SCC $[\si{US\mathdollar / tCe}]$"
            })
        elseif row == 2
            @pgf push!(sccbydiscoveryfig, {
                title = L"$T^c = %$thresholdlabel$",
                xlabel = L"Discovery $\Delta T^{\mathrm{d}}$ [\si{\degree}]"
            })
        elseif col == 1
            @pgf push!(sccbydiscoveryfig, {
                title = L"$T^c = %$thresholdlabel$",
                ylabel = L"SCC $[\si{US\mathdollar / tCe}]$"
            })
        else
            @pgf push!(sccbydiscoveryfig, {
                title = L"$T^c = %$thresholdlabel$"
            })
        end
        

        smooth!(sccbydiscovery, 3)

        curve = @pgf Plot({
            line_width = LINE_WIDTH
        }, Coordinates(figdiscoveries, sccbydiscovery))
        
        push!(sccbydiscoveryfig, curve)
        
        #= baseline = @pgf Plot({
            line_width = LINE_WIDTH,
            dashed
        }, Coordinates(figdiscoveries, fill(sccs[idx], length(figdiscoveries)))) =#
        
        #push!(sccbydiscoveryfig, baseline)
    end
    
    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "scc_bydiscovery.tikz"), sccbydiscoveryfig; include_preamble=true)
    end
    
    sccbydiscoveryfig
end

begin # SCC surface
    figdiscoveries = discoveries
    figthresholds = thresholds
    
    sccmatrix = fill(NaN, length(figthresholds), length(figdiscoveries))
    
    for (i, threshold) in enumerate(figthresholds)
        for (j, discovery) in enumerate(figdiscoveries)
            if !isnan(discoverygradient[i, j][2]) && !isnan(truegradient[i][2])
                sccmatrix[i, j] = sccs[i] * discoverygradient[i, j][2] / truegradient[i][2]
            end
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
        view = "{45}{45}",
        grid = "both",
        xmin = minimum(figdiscoveries),
        xmax = maximum(figdiscoveries),
        ymin = minimum(figthresholds),
        ymax = maximum(figthresholds),
        xtick = discoveryticks,
        xticklabels = discoverylabels,
        ytick = thresholdticks,
        yticklabels = thresholdlabels,
        y_dir = "reverse",
        width = "0.85\\textwidth",
        height = "0.85\\textwidth",
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
    }, Table([vec(repeat(figdiscoveries', length(figthresholds), 1)),
              vec(repeat(figthresholds, 1, length(figdiscoveries))),
              vec(repeat(sccs, 1, length(figdiscoveries)))]))
    
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

