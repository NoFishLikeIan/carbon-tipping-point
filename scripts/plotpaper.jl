using Revise
using UnPack
using JLD2, DotEnv, CSV
using DataFrames

using FiniteDiff
using DifferentialEquations, DifferentialEquations.EnsembleAnalysis

using KernelDensity
using Interpolations
using Plots, Printf, PGFPlotsX, Colors

using Model, Grid

include("utils/plotting.jl")
include("utils/saving.jl")

begin # Global variables
    env = DotEnv.config()

    PALETTE = color.(["#003366", "#E31B23", "#005CAB", "#DCEEF3", "#FFC325", "#E6F1EE"])
    SEQPALETTECODE = :YlOrRd
    generateseqpalette(n) = palette(SEQPALETTECODE, n + 2)[3:end]

    LINESTYLE = ["solid", "dashed", "dotted"]
    
    BASELINE_YEAR = parse(Int64, get(env, "BASELINE_YEAR", "2020"))
    DATAPATH = get(env, "DATAPATH", "data")
    PLOTPATH = get(env, "PLOTPATH", "plots")
    PRESENTATIONPATH = joinpath(PLOTPATH, "presentation")

    SAVEFIG = false 
    kelvintocelsius = 273.15
end;

begin # Import
    ΔΛ = [0., 0.06, 0.08]
    Ω = 2 * 10. .^(-4:1:-1);
	N = 21;
	domains = [
		Hogg().T₀ .+ (0., 9.),
		log.(Hogg().M₀ .* (1., 2.)),
		log.(Economy().Y₀ .* (0.5, 2.))
	]

	preferences = EpsteinZin()
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))

	models = ModelInstance[]
	

	for Δλ ∈ ΔΛ, ωᵣ ∈ Ω
		economy = Economy(ωᵣ = ωᵣ)
	    albedo = Albedo(λ₂ = 0.31 - Δλ)
		hogg = calibrateHogg(albedo)
	    model = ModelInstance(preferences, economy, hogg, albedo, calibration)

		push!(models, model)
	end
end;

G = RegularGrid(domains, N);

function stringtempdev(x::Real; digits = 2)
    fsign = x > 0 ? "+" : ""
    fmt = Printf.Format("$fsign%0.$(digits)f")
    return Printf.format(fmt, x)
end

function makedevxlabels(from, to, model::ModelInstance; step = 0.5, withcurrent = false, digits = 2)

    preindustrialx = range(from, to; step = step)
    xticks = model.hogg.Tᵖ .+ preindustrialx

    xlabels = [stringtempdev(x, digits = digits) for x in preindustrialx]

    if !withcurrent
        return (xticks, xlabels)
    end

    xlabels = [xlabels..., "\$x_0\$"]
    xticks = [xticks..., first(climate).x₀]
    idxs = sortperm(xticks)
    
    return (xticks[idxs], xlabels[idxs])
end

function generateframes(total, frames)
	step = total ÷ frames
	return [range(1, total - 2step; step = step)..., total]
end

begin # labels and axis
    TEMPLABEL = raw"Temperature deviations $T - T^{\mathrm{p}}$"
    Tspacedev = range(0., 10.; length = 51)
    Tspace = Tspacedev .+ Hogg().Tᵖ
    yearlytime = collect(0:Economy().t₁) 
    ΔTᵤ = last(Tspace) - first(Tspace)
    temperatureticks = makedevxlabels(0., ΔTᵤ, first(models); step = 1, digits = 0)
end

begin # Load IPCC data
    IPCCDATAPATH = joinpath(DATAPATH, "climate-data", "proj-median.csv")
    ipccproj = CSV.read(IPCCDATAPATH, DataFrame)

    getscenario(s::Int64) = filter(:Scenario => isequal("SSP$(s) - Baseline"), ipccproj)

    bauscenario = getscenario(5)
    seqpaletteΔλ = generateseqpalette(length(ΔΛ))
end

begin # Albedo plot
    ΔΛfig = [0, 0.06, 0.08]
    albedovariation = [(T -> Model.λ(T, Albedo(λ₂ = Albedo().λ₁ - Δλ))).(Tspace) for Δλ ∈ ΔΛfig]


    albedofig = @pgf Axis(
        {
            width = raw"1\textwidth",
            height = raw"0.6\textwidth",
            grid = "both",
            xlabel = TEMPLABEL,
            ylabel = raw"Albedo coefficient $\lambda(T)$",
            xticklabels = temperatureticks[2],
            xtick = 0:1:ΔTᵤ,
            no_markers, ultra_thick,
            xmin = 0, xmax = (ΔTᵤ)
        }
    )

    @pgf for (i, albedodata) in enumerate(albedovariation)
        curve = Plot(
            {color=seqpaletteΔλ[i], line_width="0.1cm",}, 
            Coordinates(
                collect(zip(Tspacedev, albedodata))
            )
        ) 

        legend = LegendEntry("$(ΔΛfig[i])")

        push!(albedofig, curve, legend)
    end
    @pgf albedofig["legend style"] = raw"at = {(0.3, 0.5)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "albedo.tikz"), albedofig; include_preamble = true) 
    end

    albedofig
end

begin # Nullcline plot
    nullclinevariation = Vector{Float64}[]

    for Δλ ∈ ΔΛfig
        albedo = Albedo(λ₂ = Albedo().λ₁ - Δλ)
        hogg = calibrateHogg(albedo)
        null = [Model.Mstable(T, hogg, albedo) for T ∈ Tspace]

        push!(nullclinevariation, null)
    end

    nullclinefig = @pgf Axis(
        {
            width = raw"0.7\textwidth",
            height = raw"0.7\textwidth",
            grid = "both",
            ylabel = TEMPLABEL,
            xlabel = raw"Carbon concentration $M$",
            xmin = Hogg().Mᵖ, xmax = 1200,
            xtick = 200:200:1200,
            yticklabels = temperatureticks[2],
            ytick = 0:1:ΔTᵤ,
            ymin = 0, ymax = ΔTᵤ,
            ultra_thick, 
        }
    )

    @pgf for (i, nullclinedata) in enumerate(nullclinevariation)
        coords = Coordinates(collect(zip(nullclinedata, Tspacedev)))

        curve = Plot({color = seqpaletteΔλ[i], line_width="0.1cm"}, coords) 

        legend = LegendEntry("$(ΔΛfig[i])")

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
    gcolor = first(PALETTE)

    gfig = @pgf Axis(
        {
            width = raw"0.75\linewidth",
            height = raw"0.75\linewidth",
            grid = "both",
            ylabel = raw"Growth rate $\gamma^{\mathrm{b}}$",
            xlabel = raw"Year",
            xtick = 0:20:horizon,
            xmin = 0, xmax = horizon,
            xticklabels = BASELINE_YEAR .+ (0:20:horizon),
            ultra_thick, xticklabel_style = {rotate = 45}
        }
    )   
    
    gdata = [γ(t, first(models).economy, first(models).calibration) for t ∈ yearlytime]
    coords = Coordinates(zip(yearlytime, gdata))

    markers = @pgf Plot({ only_marks, mark_options = {fill = gcolor, scale = 1.5, draw_opacity = 0}, mark_repeat = 10}, coords) 

    curve = @pgf Plot({color = gcolor, line_width="0.1cm"}, coords) 

    push!(gfig, curve, markers)

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "growthmfig.tikz"), gfig; include_preamble = true) 
    end

    gfig
end

# --- Business-as-usual dynamics
function Fbau!(du, u, model, t)

	du[1] = μ(u[1], u[2], model.hogg, model.albedo) / model.hogg.ϵ
	du[2] = γ(t, model.economy, model.calibration)
end
function Gbau!(du, u, model, t)
    

	du[1] = model.hogg.σₜ / model.hogg.ϵ
	du[2] = 0.
end

const X₀ = [Hogg().T₀, log(Hogg().M₀), log(Economy().Y₀)];

function simulatebau(Δλ; trajectories = 1000) # Business as Usual, ensemble simulation   
    k = findfirst(m -> m.albedo.λ₁ - m.albedo.λ₂ ≈ Δλ, models)
    model = models[k]
    
    prob = SDEProblem(SDEFunction(Fbau!, Gbau!), X₀[1:2], (0., 260.), model)
    
    ensemble = EnsembleProblem(prob)
    
    bausim = solve(ensemble; trajectories)
    baunullcline = (x -> Model.mstable(x, model.hogg, model.albedo)).(Tspace)
    
    return bausim, baunullcline
end

begin # Density plots
    ytick = range(2.505674517612567, 2.509962798461946; length = 10) # A bit ugly but I do not know how to remove the ticks

    densfig = @pgf Axis({
        width = raw"0.4\textwidth",
        height = raw"0.4\textwidth",
        grid = "both",
        ylabel = "Density",
        ytick = ytick, yticklabels = ["" for y ∈ ytick],
        xlabel = TEMPLABEL,
        xmin = minimum(Tspace), xmax = maximum(Tspace),
        xtick = temperatureticks[1],
        xticklabels = temperatureticks[2],
        ultra_thick, xticklabel_style = {rotate = 45}
    })

    for (i, Δλ) ∈ enumerate(ΔΛ)
        k = findfirst(m -> m.albedo.λ₁ - m.albedo.λ₂ ≈ Δλ, models)
        model = models[k]

        d = [Model.density(T, log(1.2model.hogg.M₀), model.hogg, model.albedo) for T in Tspace ]

        @pgf push!(densfig,
            Plot({
                color = seqpaletteΔλ[i],
                line_width="0.1cm", 
            }, Coordinates(zip(Tspace, d))),
            LegendEntry("\$$(Δλ)\$")
        )
    end

    @pgf densfig["legend style"] = raw"at = {(0.5, 0.4)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "densfig.tikz"), densfig; include_preamble = true)
    end
    
    densfig
end

begin # Business as usual plots
    baufig = @pgf GroupPlot(
        {
            group_style = { 
                group_size = "2 by 1", 
                horizontal_sep="0pt",
                yticklabels_at="edge left"
            }, 
            width = raw"0.6\textwidth",
            height = raw"0.6\textwidth",
            yticklabels = temperatureticks[2],
            ytick = 0:1:ΔTᵤ,
            ymin = 0, ymax = ΔTᵤ,
            xmin = Hogg().Mᵖ, xmax = 1200,
            xtick = 200:100:1100,
            grid = "both"
        }
    )

    @pgf for (i, Δλ) ∈ enumerate([0., 0.08])
        isfirst = i == 1
        Δλplots = []
        timeseriescolor = isfirst ? seqpaletteΔλ[1] : seqpaletteΔλ[end]
    
        # IPCC benchmark line
        ipccbau = @pgf Plot(
            {
                very_thick, 
                color = "black", 
                mark = "*", 
                mark_options = {scale = 1.5, draw_opacity = 0}, 
                mark_repeat = 2
            }, 
            Coordinates(zip(
                bauscenario[3:end, "CO2 concentration"], 
                bauscenario[3:end, "Temperature"]
            )))

        push!(Δλplots, ipccbau)
        push!(Δλplots, LegendEntry("IPCC"))


        # Data simulation
        bausim, baunullcline = simulatebau(Δλ; trajectories = 20)
        baumedian = [timepoint_median(bausim, t) for t in yearlytime]
        baumedianM = @. exp([u[2] for u in baumedian])
        baumedianT = @. first(baumedian) - Hogg().Tᵖ


        # Nullcline
        push!(Δλplots,
            @pgf Plot({dashed, color = "black", ultra_thick},
                Coordinates(collect(zip(exp.(baunullcline), Tspacedev)))
            )
        )

        push!(Δλplots, LegendEntry("Equilibrium"))

        mediancoords = Coordinates(zip(baumedianM, baumedianT))

        label = isfirst ? raw"$\Delta\lambda = 0$" : raw"$\Delta\lambda = 0.08$"

        @pgf begin
            push!(
                Δλplots,
                Plot({line_width="0.1cm", color = timeseriescolor, opacity = 0.8},mediancoords),
                LegendEntry(label),
                Plot({only_marks, mark_options = {fill = timeseriescolor, scale = 1.5, draw_opacity = 0, fill_opacity = 0.8}, mark_repeat = 20, forget_plot, mark = "*"}, mediancoords)
            )
        end

        @pgf for sim in bausim
            path = sim.(yearlytime)

            mpath = @. exp([u[2] for u in path])
            xpath = @. first(path) - Hogg().Tᵖ

            push!(
                Δλplots, 
                Plot({forget_plot, color = timeseriescolor, opacity = 0.2},
                    Coordinates(zip(mpath, xpath)),
                )
            )
        end
        
        nextgroup = isfirst ? {
            xlabel = raw"Carbon concentration $M$", 
            ylabel = TEMPLABEL
        } : {
            xlabel = raw"Carbon concentration $M$"
        }

        push!(baufig, nextgroup, Δλplots...)
    end

    @pgf baufig["legend style"] = raw"at = {(0.45, 0.95)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "baufig.tikz"), baufig; include_preamble = true) 
    end

    baufig
end

begin # Density
    baufig = @pgf GroupPlot(
        {
            group_style = { 
                group_size = "2 by 1", 
                horizontal_sep="0pt",
                yticklabels_at="edge left"
            }, 
            width = raw"0.6\textwidth",
            height = raw"0.6\textwidth",
            yticklabels = temperatureticks[2],
            ytick = 0:1:ΔTᵤ,
            ymin = 0, ymax = ΔTᵤ,
            xmin = Hogg().Mᵖ, xmax = 1200,
            xtick = 200:100:1100,
            grid = "both"
        }
    )

    @pgf for (i, Δλ) ∈ enumerate([0., 0.08])
        isfirst = i == 1
        Δλplots = []
        timeseriescolor = isfirst ? seqpaletteΔλ[1] : seqpaletteΔλ[end]
    
        # IPCC benchmark line
        ipccbau = @pgf Plot(
            {
                very_thick, 
                color = "black", 
                mark = "*", 
                mark_options = {scale = 1.5, draw_opacity = 0}, 
                mark_repeat = 2
            }, 
            Coordinates(zip(
                bauscenario[3:end, "CO2 concentration"], 
                bauscenario[3:end, "Temperature"]
            )))

        push!(Δλplots, ipccbau)
        
        push!(Δλplots, LegendEntry("SSP5 - Baseline"))


        # Data simulation
        bausim, baunullcline = simulatebau(Δλ; trajectories = 20)
        baumedian = [timepoint_median(bausim, t) for t in yearlytime]
        baumedianM = @. exp([u[2] for u in baumedian])
        baumedianT = @. first(baumedian) - Hogg().Tᵖ


        # Nullcline
        push!(Δλplots,
            @pgf Plot({dashed, color = "black", ultra_thick, forget_plot},
                Coordinates(collect(zip(exp.(baunullcline), Tspacedev)))
            )
        )

        mediancoords = Coordinates(zip(baumedianM, baumedianT))

        label = isfirst ? raw"$\Delta \lambda = 0$" : raw"$\Delta \lambda = 0.08$"

        @pgf begin
            push!(
                Δλplots,
                Plot({line_width="0.1cm", color = timeseriescolor, opacity = 0.8},mediancoords),
                LegendEntry(label),
                Plot({only_marks, mark_options = {fill = timeseriescolor, scale = 1.5, draw_opacity = 0, fill_opacity = 0.8}, mark_repeat = 20, forget_plot, mark = "*"}, mediancoords)
            )
        end

        @pgf for sim in bausim
            path = sim.(yearlytime)

            mpath = @. exp([u[2] for u in path])
            xpath = @. first(path) - Hogg().Tᵖ
            
            push!(
                Δλplots, 
                Plot({forget_plot, color = timeseriescolor, opacity = 0.2},
                    Coordinates(zip(mpath, xpath)),
                )
            )
        end
        
        nextgroup = isfirst ? {
            xlabel = raw"Carbon concentration $M$", 
            ylabel = TEMPLABEL
        } : {
            xlabel = raw"Carbon concentration $M$"
        }

        push!(baufig, nextgroup, Δλplots...)
    end

    @pgf baufig["legend style"] = raw"at = {(0.6, 0.95)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "baufig.tikz"), baufig; include_preamble = true) 
    end

    baufig
end

begin # Carbon decay calibration
    sinkspace = range(Hogg().N₀, 1.2 * Hogg().N₀; length = 101)
    decay = (n -> Model.δₘ(n, Hogg())).(sinkspace)

    decayfig = @pgf Axis(
        {
            width = raw"0.7\textwidth",
            height = raw"0.5\textwidth",
            grid = "both",
            xlabel = raw"Carbon stored in sinks $N$",
            ylabel = raw"Decay of CO$_2$ in the atmosphere $\delta_m$",
            no_markers,
            ultra_thick,
            xmin = first(sinkspace), xmax = last(sinkspace),
            ymin = 0
        }
    )

    @pgf push!(decayfig, 
        Plot(
            { color = PALETTE[1], ultra_thick }, 
            Coordinates(
                collect(zip(sinkspace, decay))
            )
        ))
    

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "decay.tikz"), decayfig; include_preamble = true) 
    end

    decayfig
end

begin # Carbon decay path
    bausim, baunullcline = simulatebau(0.; trajectories = 1)
    M = exp.([u[2] for u in bausim[1].u])
    decaysim = Model.δₘ.(M, Ref(Hogg()))
    
    Msparse = exp.([bausim(y)[1][2] for y in 0:10:horizon])
    decaysimsparse = Model.δₘ.(Msparse, Ref(Hogg()))

    decaypathfig = @pgf Axis(
        {
            width = raw"0.7\textwidth",
            height = raw"0.5\textwidth",
            grid = "both",
            xlabel = raw"Carbon concentration $M$",
            ylabel = raw"Decay of CO$_2$ in the atmosphere $\delta_m$",
            no_markers,
            ultra_thick,
            xmin = Hogg().M₀, xmax = maximum(M),
            ymin = -1e-3
        }
    )

    @pgf push!(decaypathfig,
        Plot({ color = PALETTE[1], ultra_thick }, Coordinates(M, decaysim))
    )

    decayscatter = @pgf Plot({
        very_thick, 
        color = "black", 
        mark = "*", only_marks,
        mark_options = {scale = 1.5, draw_opacity = 0}
    }, Coordinates(Msparse, decaysimsparse))

    @pgf push!(decaypathfig, decayscatter)


    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "decaypathfig.tikz"), decaypathfig; include_preamble = true) 
    end

    decaypathfig
end

begin # Damage fig
    damagefig = @pgf Axis(
        {
            width = raw"0.5\textwidth",
            height = raw"0.5\textwidth",
            grid = "both",
            xlabel = TEMPLABEL,
            ylabel = raw"Damage function $d(T)$",
            xticklabels = temperatureticks[2],
            xtick = 0:1:ΔTᵤ,
            no_markers,
            ultra_thick,
            xmin = 0, xmax = ΔTᵤ,
            ytick = 0:0.1:1, xticklabel_style = {rotate = 45}
        }
    )

    @pgf damagecurve = Plot({color = "black", line_width = "0.1cm"},
        Coordinates(Tspacedev, [Model.d(T, Economy(), Hogg()) for T in Tspace])
    )

    push!(damagefig, damagecurve)

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "damagefig.tikz"), damagefig; include_preamble = true) 
    end

    damagefig
end

# --- Optimal emissions 
#FIXME: The script below is out of date
results = loadtotal(models, G; datapath = DATAPATH);

begin
	ΔT, Δm, Δy = G.domains
	spacenodes = ntuple(i -> range(G.domains[i]...; length = N), 3)

	interpolations = []

	for (k, res) in enumerate(results)
		ts, V, policy = res
			
		nodes = (spacenodes..., ts)
		χitp = linear_interpolation(nodes, first.(policy); extrapolation_bc = Flat())
		αitp = linear_interpolation(nodes, last.(policy); extrapolation_bc = Flat())
		Vitp = linear_interpolation(nodes, V; extrapolation_bc = Flat())

		push!(interpolations, (χitp, αitp, Vitp))
	end
end;

function F!(dx, x, parameters, t)
    model, interpolation = parameters
    χitp, αitp, _ = interpolation

	T, m, y = x
	
	χ = χitp(T, m, y, t)
	α = αitp(T, m, y, t)
	
	dx[1] = μ(T, m, model.hogg, model.albedo) / model.hogg.ϵ
	dx[2] = γ(t, model.economy, model.calibration) - α
	dx[3] = b(t, Point(T, m, y), Policy(χ, α), model)

    return
end;

function G!(dx, x, parameters, t)
    model = first(parameters)

	dx[1] = model.hogg.σₜ / model.hogg.ϵ
	dx[2] = 0.
	dx[3] = model.economy.σₖ
	
	return
end;

begin
	tspan = (0., Economy().t₁)

    fn = SDEFunction(F!, G!)

	problems = [
        SDEProblem(fn, X₀, tspan, params) 
        for params in zip(models, interpolations)
    ]
    
    solutions = [solve(EnsembleProblem(prob), EnsembleDistributed(); trajectories = 20) for prob ∈ problems]
end;

begin # Temperature simulation
    simspan = 2020 .+ tspan
    simtime = range(simspan...; step = 5)
    
    simfig = @pgf GroupPlot(
        {
            group_style = { 
                group_size = "1 by 3", 
                vertical_sep="0pt",
                xticklabels_at="edge bottom"
            }, 
            width = raw"\textwidth",
            height = raw"0.45\textwidth",
            xmin = 2030, xmax = simspan[2],
            xtick = range(simspan...; step = 10),
            grid = "both"
        }
    )

    Mfig = []
    εfig = []

    @pgf for (i, Δλ) ∈ enumerate([0., 0.08])
        solution = solutions[i]
        zerotime = simtime .- simspan[1]

        timeseriescolor = i > 1 ? seqpaletteΔλ[end] : seqpaletteΔλ[1]
    
        # Data simulation
        median = [timepoint_median(solution, t) for t in zerotime]
        M = @. exp([u[2] for u in median])
        ΔT = @. first(median) - model.hogg.Tᵖ

        epath = [Model.ε(tᵢ, exp(u[2]), αitp(u..., Δλ, tᵢ), model) for (tᵢ, u) in zip(zerotime, median)]

        label = "\$\\Delta \\lambda = $Δλ\$"

        @pgf begin
            push!(Tfig,
                Plot({ ultra_thick, color = timeseriescolor, opacity = 0.8, line_width="0.1cm" },
                    Coordinates(zip(simtime, ΔT))
                ), LegendEntry(label))
            
            push!(Mfig,
                Plot(
                    { ultra_thick, color = timeseriescolor, opacity = 0.8, line_width="0.1cm" },
                    Coordinates(zip(simtime, M))))

            push!(εfig,
                Plot({ ultra_thick, color = timeseriescolor, opacity = 0.8, line_width="0.1cm" }, 
                Coordinates(zip(simtime, epath)))
            )
        end

        @pgf for sim in solution
            path = sim.(zerotime)

            Mᵢ = @. exp([u[2] for u in path])
            ΔTᵢ = @. first(path) - model.hogg.Tᵖ

            epathᵢ = [Model.ε(tᵢ, exp(u[2]), αitp(u..., Δλ, tᵢ), model) for (tᵢ, u) in zip(zerotime, median)]

            push!(
                Tfig, 
                Plot({forget_plot, color = timeseriescolor, opacity = 0.1},
                    Coordinates(zip(simtime, ΔTᵢ)),
                )
            )
            push!(
                Mfig, 
                Plot({forget_plot, color = timeseriescolor, opacity = 0.1},
                    Coordinates(zip(simtime, Mᵢ)),
                )
            )
            push!(
                εfig, 
                Plot({forget_plot, color = timeseriescolor, opacity = 0.1},
                    Coordinates(zip(simtime, epathᵢ)),
                )
            )
        end
        end

    @pgf push!(simfig,
        { ylabel = TEMPLABEL }, Tfig..., 
        { ylabel = "Carbon Concentration \$M\$" }, Mfig...,
        { ylabel = "Fraction of abated emissions \$\\varepsilon \$", xlabel = "Year" }, εfig...
    )

    @pgf simfig["legend style"] = raw"at = {(0.24, 0.95)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "simfig.tikz"), simfig; include_preamble = true) 
    end

    simfig
end

begin # SCC simulation
    simspan = 2020 .+ (0., 100.)
    simtime = range(simspan...; step = 1)
    
    sccfig = @pgf Axis(
        { 
            width = raw"\textwidth",
            height = raw"0.6\textwidth",
            xmin = simspan[1], xmax = simspan[2],
            xtick = range(simspan...; step = 10),
            grid = "both"
        }
    )

    @pgf for (i, Δλ) ∈ enumerate([0., 0.08])
        solution = solutions[i]

        timeseriescolor = i > 1 ? seqpaletteΔλ[end] : seqpaletteΔλ[1]
    
        # Data simulation
        sccpath = [scc(timepoint_median(solution, tᵢ), Δλ, tᵢ) for tᵢ in (simtime .- simspan[1])]
        label = "\$\\Delta \\lambda = $Δλ\$"

        @pgf push!(sccfig,
            Plot(
                { ultra_thick, color = timeseriescolor, opacity = 0.8 },
                Coordinates(zip(simtime, sccpath))
            ), LegendEntry(label))
    end

    @pgf sccfig["legend style"] = raw"at = {(0.24, 0.95)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "sccfig.tikz"), sccfig; include_preamble = true) 
    end

    sccfig
end