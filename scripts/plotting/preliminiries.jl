using Revise
using JLD2, DotEnv, CSV
using UnPack
using DataFrames, DataStructures

using FiniteDiff
using DifferentialEquations, DifferentialEquations.EnsembleAnalysis

using KernelDensity
using Interpolations
using Plots, Printf, PGFPlotsX, Colors, ColorSchemes
using Statistics

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")

using Model, Grid

includet("../utils/saving.jl")
includet("../utils/simulating.jl")
includet("utils.jl")

begin # Global variables
    env = DotEnv.config()
    BASELINE_YEAR = 2020
    DATAPATH = get(env, "DATAPATH", "data")
    PLOTPATH = get(env, "PLOTPATH", "plots")
    PRESENTATIONPATH = joinpath(PLOTPATH, "presentation")

    SAVEFIG = false 
    kelvintocelsius = 273.15
    LINE_WIDTH = 2.5
end;

begin # Construct models and grids
    ΔΛ = [0., 0.06, 0.08];
	N = 51;

	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	preferences = EpsteinZin()
    damages = GrowthDamages()
    economy = Economy()
    jump = Jump()

    jumpmodel = JumpModel(jump, preferences, damages, economy, Hogg(), calibration)
    
	models = TippingModel[]
	for Δλ ∈ ΔΛ
	    albedo = Albedo(λ₂ = Albedo().λ₁ - Δλ)
		hogg = calibrateHogg(albedo)

	    model = TippingModel(albedo, preferences, damages, economy, hogg, calibration)

		push!(models, model)
	end
end;

begin # Labels, colors and axis
    PALETTE = colorschemes[:grays]
    graypalette = n -> get(PALETTE, range(0.1, 0.8; length = n)) |> reverse
    
    λgrays = graypalette(length(ΔΛ))
    λcolors = Dict(ΔΛ .=> graypalette(length(ΔΛ)))


    TEMPLABEL = "Temperature deviations \$T - T^{p}\$"

    ΔTmax = 9.
    ΔTspace = range(0., ΔTmax; length = 51)
    Tspace = ΔTspace .+ Hogg().Tᵖ

    horizon = round(Int64, calibration.tspan[2])
    yearlytime = 0:1:horizon

    temperatureticks = collect.(makedeviationtickz(0., ΔTmax, first(models); step = 1, digits = 0))

    Tmin, Tmax = extrema(temperatureticks[1])
end

begin # Load IPCC data
    IPCCDATAPATH = joinpath(DATAPATH, "climate-data", "proj-median.csv")
    ipccproj = CSV.read(IPCCDATAPATH, DataFrame)

    getscenario(s) = filter(:Scenario => isequal("SSP$(s) - Baseline"), ipccproj)

    bauscenario = getscenario(5)
end

begin # Albedo plot
    albedovariation = [(T -> Model.λ(T, Albedo(λ₂ = Albedo().λ₁ - Δλ))).(Tspace) for Δλ ∈ ΔΛ]

    albedofig = @pgf Axis(
        {
            width = raw"0.5\textwidth",
            height = raw"0.5\textwidth",
            grid = "both",
            xlabel = TEMPLABEL,
            ylabel = "\$\\lambda(T)\$",
            xticklabels = temperatureticks[2],
            xtick = temperatureticks[1],
            xmin = Tmin, xmax = Tmax,
            ymin = 0.2, ymax = 0.32,
            ytick = 0.2:0.05:0.35
        }
    )

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "skeleton-albedo.tikz"), albedofig; include_preamble = true) 
    end

    @pgf for (Δλ, loss) in zip(ΔΛ, albedovariation)
        curve = Plot(
            { color = λcolors[Δλ], line_width = LINE_WIDTH }, 
            Coordinates(zip(Tspace, loss))
        ) 

        label = @sprintf("%.0f \\%%", 100 * Δλ)
        legend = LegendEntry(label)

        push!(albedofig, curve, legend)
    end

    @pgf albedofig["legend style"] = raw"at = {(0.4, 0.4)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "albedo.tikz"), albedofig; include_preamble = true) 
    end

    albedofig
end

begin # Nullcline plot
    nullclinevariation = Vector{Float64}[]

    for (k, Δλ) ∈ enumerate(ΔΛ)
        model = models[k]
        null = [Model.Mstable(T, model.hogg, model.albedo) for T in Tspace]

        push!(nullclinevariation, null)
    end

    nullclinefig = @pgf Axis(
        {
            width = raw"0.7\textwidth",
            height = raw"0.7\textwidth",
            grid = "both",
            ylabel = TEMPLABEL,
            xlabel = raw"Carbon concentration $M$",
            xmin = defaultmodel.hogg.Mᵖ, xmax = 900,
            xtick = 200:100:900,
            yticklabels = temperatureticks[2],
            ytick = temperatureticks[1],
            ymin = Tmin, ymax = Tmax
        }
    )

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "skeleton-nullcline.tikz"), nullclinefig; include_preamble = true) 
    end

    @pgf for (Δλ, nullclinedata) in zip(ΔΛ, nullclinevariation)
        nullclinecoords = Coordinates(zip(nullclinedata, Tspace))

        curve = Plot({color = λcolors[Δλ], line_width=LINE_WIDTH}, nullclinecoords) 

        label = @sprintf("%.0f \\%%", 100 * Δλ)
        legend = LegendEntry(label)

        push!(nullclinefig, curve, legend)
    end

    @pgf nullclinefig["legend style"] = raw"at = {(0.95, 0.3)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "nullcline.tikz"), nullclinefig; include_preamble = true) 
    end

    nullclinefig
end

begin # Growth of carbon concentration
    horizon = Int(last(yearlytime))

    gfig = @pgf Axis(
        {
            width = raw"0.75\linewidth",
            height = raw"0.75\linewidth",
            grid = "both",
            ylabel = raw"Growth rate $\gamma^{b}$",
            xlabel = raw"Year",
            xtick = 0:20:horizon,
            xmin = 0, xmax = horizon,
            xticklabels = BASELINE_YEAR .+ (0:20:horizon),
            ultra_thick, xticklabel_style = {rotate = 45}
        }
    )   
    
    gdata = [γ(t, calibration) for t ∈ yearlytime]
    coords = Coordinates(zip(yearlytime, gdata))

    markers = @pgf Plot({ only_marks, mark_options = {fill = "black", scale = 1.5, draw_opacity = 0}, mark_repeat = 10}, coords) 

    curve = @pgf Plot({color = "black", line_width="0.1cm"}, coords) 

    push!(gfig, curve, markers)

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "growthmfig.tikz"), gfig; include_preamble = true) 
    end

    gfig
end

# --- Business-as-usual dynamics
begin # Density plots
    ytick = range(2.505674517612567, 2.509962798461946; length = 10) # A bit ugly but I do not know how to remove the ticks

    densfig = @pgf Axis({
        width = raw"0.4\textwidth",
        height = raw"0.4\textwidth",
        grid = "both",
        ylabel = "Density",
        ytick = ytick, yticklabels = ["" for y ∈ ytick],
        xlabel = TEMPLABEL,
        xmin = Tmin, xmax = Tmax,
        xtick = temperatureticks[1],
        xticklabels = temperatureticks[2],
        ultra_thick, xticklabel_style = {rotate = 45}
    })

    for (k, Δλ) ∈ enumerate(ΔΛ)
        model = models[k]
        d = [Model.density(T, log(1.2model.hogg.M₀), model.hogg, model.albedo) for T in Tspace ]
        
        l, u = extrema(d)
        d = @. (d - l) / (u - l)

        label = @sprintf("%.0f \\%%", 100 * Δλ)
        legend = LegendEntry(label)
        @pgf push!(densfig,
            Plot({
                color = λcolors[Δλ],
                line_width="0.1cm", 
            }, Coordinates(zip(Tspace, d))),
            legend
        )
    end

    @pgf densfig["legend style"] = raw"at = {(0.4, 0.4)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "densfig.tikz"), densfig; include_preamble = true)
    end
    
    densfig
end

const baufn = SDEFunction(Fbau!, G!);

function simulatebau(model::TippingModel; trajectories = 100, X₀ = [model.hogg.T₀, log(model.hogg.M₀)])
    prob = SDEProblem(baufn, X₀, (0., 80.), model)
    ensemble = EnsembleProblem(prob)
    
    sol = solve(ensemble; trajectories)

    return sol
end

function simulatebau(model::JumpModel; trajectories =  100, X₀ = [model.hogg.T₀, log(model.hogg.M₀)])
    diffprob = SDEProblem(baufn, X₀, (0., 80.), model)
    varjump = VariableRateJump(rate, affect!)
    prob = JumpProblem(diffprob, Direct(), varjump)

    ensemble = EnsembleProblem(prob)
    
    solve(ensemble, SRIW1(); trajectories)
end

bautime = 0:80

begin # Single simulation
    
end

begin # Side by side BAU Δλ = 0.8
    baumodels = models

    yeartickstep = 10

    baufig = @pgf GroupPlot(
        {
            group_style = { 
                group_size = "1 by $(length(baumodels))", 
                vertical_sep="1pt",
                xticklabels_at="edge bottom"
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
        PGFPlotsX.save(joinpath(PLOTPATH, "baufig.tikz"), baufig; include_preamble = true) 
    end    

    for (i, model) in enumerate(baumodels)
        nextgroup = @pgf if i == 1
                 { 
                    ylabel = TEMPLABEL,             
                    yticklabels = temperatureticks[2],
                    ytick = temperatureticks[1]
                 }
            elseif i == length(baumodels)
                {
                    ylabel = TEMPLABEL,
                    xlabel = "Year",      
                    yticklabels = temperatureticks[2][1:(end - 1)],
                    ytick = temperatureticks[1][1:(end - 1)]
                }
            else
                {
                    ylabel = TEMPLABEL,
                    yticklabels = temperatureticks[2][1:(end - 1)],
                    ytick = temperatureticks[1][1:(end - 1)]
                }
            end

        push!(baufig, nextgroup)

        bausim = simulatebau(model; trajectories = 10_000)
        medianpath = first.(timeseries_point_median(bausim, bautime).u)
        lower = first.(timeseries_point_quantile(bausim, 0.05, bautime).u)
        upper = first.(timeseries_point_quantile(bausim, 0.95, bautime).u)

        medianplot = @pgf Plot({ line_width = LINE_WIDTH }, Coordinates(zip(bautime, medianpath)) )
        
        lowerplot = @pgf Plot({ draw = "none", "name path=lower" }, Coordinates(zip(bautime, lower)) )
        upperplot = @pgf Plot({ draw = "none", "name path=upper" }, Coordinates(zip(bautime, upper)) )
        confidencebands = @pgf Plot({ fill = "gray", opacity = 0.4 }, raw"fill between [of=lower and upper]")

        push!(baufig, lowerplot, upperplot, confidencebands, medianplot)
    
    end

    @pgf baufig["legend style"] = raw"at = {(0.45, 0.95)}"


    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "baufig.tikz"), baufig; include_preamble = true) 
    end

    baufig
end

begin # Carbon decay calibration
    tippingmodel = last(models)

    sinkspace = range(tippingmodel.hogg.N₀, 1.2 * tippingmodel.hogg.N₀; length = 51)
    decay = [Model.δₘ(n, tippingmodel.hogg) for n in sinkspace]

    decayfig = @pgf Axis(
        {
            width = raw"0.7\textwidth",
            height = raw"0.5\textwidth",
            grid = "both",
            xlabel = raw"Carbon stored in sinks $N$",
            ylabel = raw"Decay of CO$_2$ in the atmosphere $\delta_m$",
            xmin = first(sinkspace), xmax = last(sinkspace),
            ymin = 0
        }
    )

    @pgf push!(decayfig, Plot({ line_width = LINE_WIDTH }, Coordinates(zip(sinkspace, decay))))
    

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "decay.tikz"), decayfig; include_preamble = true) 
    end

    decayfig
end

begin # Carbon decay path
    bausim = simulatebau(tippingmodel)
    bauM = exp.(last.(timeseries_point_median(bausim, bautime).u))
    mediandecay = [Model.δₘ(M, tippingmodel.hogg) for M in bauM]

    decaypathfig = @pgf Axis(
        {
            width = raw"0.7\textwidth",
            height = raw"0.5\textwidth",
            grid = "both",
            xlabel = raw"Carbon concentration $M$",
            ylabel = raw"Decay of CO$_2$ in the atmosphere $\delta_m$",
            xmin = tippingmodel.hogg.M₀, xmax = 600
        }
    )

    @pgf push!(decaypathfig,
        Plot({ line_width = LINE_WIDTH }, Coordinates(bauM, mediandecay))
    )


    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "decaypathfig.tikz"), decaypathfig; include_preamble = true) 
    end

    decaypathfig
end

begin # Damage fig

    ds = [Model.d(T, damages, tippingmodel.hogg) for T in Tspace]

    maxpercentage = ceil(maximum(ds), digits = 2)
    ytick = 0:0.02:maxpercentage
    yticklabels = [@sprintf("%.0f \\%%", 100 * y) for y in ytick]

    damagefig = @pgf Axis(
        {
            width = raw"0.5\textwidth",
            height = raw"0.5\textwidth",
            grid = "both",
            xlabel = TEMPLABEL,
            ylabel = raw"Damage function $d(T)$",
            xticklabels = temperatureticks[2],
            xtick = temperatureticks[1],
            xmin = Tmin, xmax = Tmax,
            xticklabel_style = {rotate = 45},
            yticklabels = yticklabels, ytick = ytick, ymin = -0.01
        }
    )

    ds = [Model.d(T, damages, tippingmodel.hogg) for T in Tspace]

    @pgf damagecurve = Plot({line_width = LINE_WIDTH},
        Coordinates(Tspace, ds)
    )

    push!(damagefig, damagecurve)

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "damagefig.tikz"), damagefig; include_preamble = true) 
    end

    damagefig
end
