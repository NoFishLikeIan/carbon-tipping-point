using Revise
using UnPack
using JLD2, DotEnv, CSV
using DataFrames
using DataStructures

using FiniteDiff
using DifferentialEquations, DifferentialEquations.EnsembleAnalysis

using KernelDensity
using Interpolations
using Plots, Printf, PGFPlotsX, Colors
using Statistics

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
    ΔΛ = [0.06, 0.08];
    Ω = [0.002]
	N = 23;
	domains = [
		Hogg().T₀ .+ (0., 9.),
		log.(Hogg().M₀ .* (1., 2.)),
		log.(Economy().Y₀ .* (0.5, 2.))
	]

	preferences = EpsteinZin()
	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
    damages = GrowthDamages(ξ = 0.000266, υ = 3.25)

	models = ModelInstance[]
    jumpmodels = ModelBenchmark[]
	

	for Δλ ∈ ΔΛ, ωᵣ ∈ Ω
		economy = Economy(ωᵣ = ωᵣ)
	    albedo = Albedo(λ₂ = 0.31 - Δλ)
		hogg = calibrateHogg(albedo)

	    model = ModelInstance(preferences, economy, damages, hogg, albedo, calibration)
        jumpmodel = ModelBenchmark(preferences, economy, damages, Hogg(), Jump(), calibration)

		push!(models, model)
        push!(jumpmodels, jumpmodel)
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
    seqpaletteΔλ = Dict(ΔΛ .=> generateseqpalette(length(ΔΛ)))
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
            xmin = 0, xmax = (ΔTᵤ),
            ymin = 0.2, ymax = 0.32
        }
    )

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "skeleton-albedo.tikz"), albedofig; include_preamble = true) 
    end

    @pgf for (Δλ, albedodata) in zip(ΔΛfig, albedovariation)
        curve = Plot(
            {color=seqpaletteΔλ[Δλ], line_width="0.1cm",}, 
            Coordinates(
                collect(zip(Tspacedev, albedodata))
            )
        ) 

        label = @sprintf("%.0f \\%%", 100 * Δλ)
        legend = LegendEntry(label)

        push!(albedofig, curve, legend)
    end
    @pgf albedofig["legend style"] = raw"at = {(0.3, 0.5)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "albedo.tikz"), albedofig; include_preamble = true) 
    end

    albedofig
end

begin # Nullcline plot
    nullclineTspace = range(Hogg().Tᵖ .+ (0., 7.)...; length = length(Tspace))
    nullclinevariation = Vector{Float64}[]

    for Δλ ∈ ΔΛfig
        albedo = Albedo(λ₂ = Albedo().λ₁ - Δλ)
        hogg = calibrateHogg(albedo)
        null = [Model.Mstable(T, hogg, albedo) for T ∈ nullclineTspace]

        push!(nullclinevariation, null)
    end

    nullclinefig = @pgf Axis(
        {
            width = raw"0.7\textwidth",
            height = raw"0.7\textwidth",
            grid = "both",
            ylabel = TEMPLABEL,
            xlabel = raw"Carbon concentration $M$",
            xmin = Hogg().Mᵖ, xmax = 900,
            xtick = 200:100:900,
            yticklabels = temperatureticks[2],
            ytick = 0:1:ΔTᵤ,
            ymin = 0, ymax = ΔTᵤ,
            ultra_thick, 
        }
    )

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "skeleton-nullcline.tikz"), nullclinefig; include_preamble = true) 
    end

    @pgf for (Δλ, nullclinedata) in zip(ΔΛfig, nullclinevariation)
        coords = Coordinates(collect(zip(nullclinedata, Tspacedev)))

        curve = Plot({color = seqpaletteΔλ[Δλ], line_width="0.1cm"}, coords) 

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

    for Δλ ∈ ΔΛfig
        k = findfirst(m -> m.albedo.λ₁ - m.albedo.λ₂ ≈ Δλ, models)
        model = models[k]

        d = [Model.density(T, log(1.2model.hogg.M₀), model.hogg, model.albedo) for T in Tspace ]
        
        l, u = extrema(d)
        d = @. (d - l) / (u - l)

        label = @sprintf("%.0f \\%%", 100 * Δλ)
        legend = LegendEntry(label)
        @pgf push!(densfig,
            Plot({
                color = seqpaletteΔλ[Δλ],
                line_width="0.1cm", 
            }, Coordinates(zip(Tspace, d))),
            legend
        )
    end

    @pgf densfig["legend style"] = raw"at = {(0.5, 0.4)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "densfig.tikz"), densfig; include_preamble = true)
    end
    
    densfig
end

function Fbau!(du, u, model::ModelInstance, t)
	du[1] = μ(u[1], u[2], model.hogg, model.albedo) / model.hogg.ϵ
	du[2] = γ(t, model.economy, model.calibration)
end
function Fbau!(du, u, model::ModelBenchmark, t)
    du[1] = μ(u[1], u[2], model.hogg) / model.hogg.ϵ
	du[2] = γ(t, model.economy, model.calibration)
end
function Gbau!(du, u, model, t)    
	du[1] = model.hogg.σₜ / model.hogg.ϵ
	du[2] = 0.
end

function rate(u, model, t)
    intensity(u[1], model.hogg, model.jump)
end
function affect!(integrator)
    model = integrator.p
    integrator.u[1] += 0.5 # increase(integrator.u[1], model.hogg, model.jump)
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

function simulatebaujump(; trajectories = 1000)
    model = first(jumpmodels)

    diffprob = SDEProblem(SDEFunction(Fbau!, Gbau!), X₀[1:2], (0., 260.), model)
    varjump = VariableRateJump(rate, affect!)
    prob = JumpProblem(diffprob, Direct(), varjump)

    ensemble = EnsembleProblem(prob)
    
    bausim = solve(ensemble, SRIW1(); trajectories)
    baunullcline = [Model.mstable(T, model.hogg) for T ∈ Tspace]
    
    return bausim, baunullcline

end

begin # Side by side BAU plots
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

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "baufig.tikz"), baufig; include_preamble = true) 
    end    

    @pgf for (i, Δλ) ∈ enumerate([0., 0.08])
        isfirst = Δλ ≈ 0.
        Δλplots = []
        timeseriescolor = seqpaletteΔλ[Δλ]
    
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

        label = @sprintf("\$\\Delta\\lambda = %.0f \\%%\$", 100 * Δλ)

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

    baufig
end

begin # Individual BAU plots
    Δλ = 0.06
    baufig = @pgf Axis(
        {
            width = raw"\linewidth",
            height = raw"0.8\linewidth",
            ylabel = TEMPLABEL,
            xlabel = raw"Carbon concentration $M$",
            yticklabels = temperatureticks[2],
            ytick = 0:1:ΔTᵤ,
            ymin = 0, ymax = ΔTᵤ,
            xmin = Hogg().Mᵖ, xmax = 1200,
            xtick = 200:100:1300,
            grid = "both"
        }
    )
    
    # IPCC benchmark line
    ipcccoords = Coordinates(zip(bauscenario[3:end, "CO2 concentration"], bauscenario[3:end, "Temperature"]))

    ipccbau = @pgf Plot({
        very_thick, 
        color = "black", 
        mark = "*", 
        mark_options = {scale = 1.5, draw_opacity = 0}, 
        mark_repeat = 2
    }, ipcccoords)

    push!(baufig, LegendEntry("SSP5 - Baseline"), ipccbau)

    # Data simulation
    bausim, baunullcline = simulatebau(Δλ; trajectories = 30);
    baumedian = timeseries_point_median(bausim, yearlytime)
    baumedianM = @. exp(last(baumedian.u))
    baumedianT = @. first(baumedian.u) - Hogg().Tᵖ

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "skeleton-baufig.tikz"), baufig; include_preamble = true) 
    end

    # Nullcline
    push!(baufig, LegendEntry(raw"$\mu(T, M) = 0$"),
        @pgf Plot({dashed, color = "black", ultra_thick},
            Coordinates(collect(zip(exp.(baunullcline), Tspacedev))))
    )

    mediancoords = Coordinates(zip(baumedianM, baumedianT))

    label = @sprintf("\$\\Delta\\lambda = %.0f \\%%\$", 100 * Δλ)

    @pgf push!(
        baufig,
        Plot({ line_width="0.1cm", color = seqpaletteΔλ[Δλ] }, mediancoords),
        LegendEntry(label),
        Plot({ only_marks, mark_options = { fill = seqpaletteΔλ[Δλ], scale = 1.5, draw_opacity = 0 }, mark_repeat = 20, forget_plot, mark = "*"}, mediancoords)
    )

    @pgf for sim in bausim
        path = sim.(yearlytime)

        mpath = @. exp([u[2] for u in path])
        xpath = @. first(path) - Hogg().Tᵖ

        push!(
            baufig, 
            Plot({forget_plot, color = seqpaletteΔλ[Δλ], opacity = 0.2},
                Coordinates(zip(mpath, xpath)),
            )
        )
    end

    @pgf baufig["legend style"] = raw"at = {(0.4, 0.975)}"


    if SAVEFIG
        filelabel = @sprintf("%.0f", 100 * Δλ)
        PGFPlotsX.save(joinpath(PLOTPATH, "baufig_$filelabel.tikz"), baufig; include_preamble = true) 
    end

    baufig
end

begin # Jump process
    baufig = @pgf Axis({
            width = raw"\linewidth",
            height = raw"0.8\linewidth",
            ylabel = TEMPLABEL,
            xlabel = raw"Carbon concentration $M$",
            yticklabels = temperatureticks[2],
            ytick = 0:1:ΔTᵤ,
            ymin = 0, ymax = ΔTᵤ,
            xmin = Hogg().Mᵖ, xmax = 1200,
            xtick = 200:100:1300,
            grid = "both"
        })
    
    # IPCC benchmark line
    ipcccoords = Coordinates(zip(bauscenario[3:end, "CO2 concentration"], bauscenario[3:end, "Temperature"]))

    ipccbau = @pgf Plot({
        very_thick, 
        color = "black", 
        mark = "*", 
        mark_options = {scale = 1.5, draw_opacity = 0}, 
        mark_repeat = 2
    }, ipcccoords)

    push!(baufig, LegendEntry("SSP5 - Baseline"), ipccbau)

    # Data simulation
    bausim, baunullcline = simulatebaujump(trajectories = 30);
    baumedian = timeseries_point_median(bausim, yearlytime)
    baumedianM = @. exp(last(baumedian.u))
    baumedianT = @. first(baumedian.u) - Hogg().Tᵖ

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "skeleton-baufig.tikz"), baufig; include_preamble = true) 
    end

    # Nullcline
    push!(baufig, LegendEntry(raw"$\mu(T, M) = 0$"),
        @pgf Plot({dashed, color = "black", ultra_thick},
            Coordinates(collect(zip(exp.(baunullcline), Tspacedev))))
    )

    mediancoords = Coordinates(zip(baumedianM, baumedianT))

    label = "Stochastic"

    jumpcolor = RGB(0, 77 / 255, 64 / 255)

    @pgf push!(
        baufig,
        Plot({ line_width="0.1cm", color = jumpcolor }, mediancoords),
        LegendEntry(label),
        Plot({ only_marks, mark_options = { fill = jumpcolor, scale = 1.5, draw_opacity = 0 }, mark_repeat = 20, forget_plot, mark = "*"}, mediancoords)
    )

    @pgf for sim in bausim
        path = sim.(yearlytime)

        mpath = @. exp([u[2] for u in path])
        xpath = @. first(path) - Hogg().Tᵖ

        push!(
            baufig, 
            Plot({forget_plot, color = jumpcolor, opacity = 0.2},
                Coordinates(zip(mpath, xpath)),
            )
        )
    end

    @pgf baufig["legend style"] = raw"at = {(0.4, 0.975)}"


    if SAVEFIG
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
            xticklabel_style = {rotate = 45}
        }
    )

    ds = [Model.d(T, damages, Hogg()) for T in Tspace]

    @pgf damagecurve = Plot({color = "black", line_width = "0.1cm"},
        Coordinates(Tspacedev, ds)
    )

    push!(damagefig, damagecurve)

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "damagefig.tikz"), damagefig; include_preamble = true) 
    end

    damagefig
end

# --- Optimal emissions 
begin # Load simulation
    results = loadtotal(models, G; datapath = DATAPATH)
	jumpresults = loadtotal(jumpmodels, G; datapath = DATAPATH)
end;

begin # Construct interpolations
	ΔT, Δm, Δy = G.domains
	spacenodes = ntuple(i -> range(G.domains[i]...; length = N), 3)

	resultsmap = OrderedDict()
	jumpresultsmap = OrderedDict()

	for ωᵣ ∈ Ω
		j = findfirst(m -> ωᵣ ≈ m.economy.ωᵣ, jumpmodels)

		res = jumpresults[j]	
		ts, V, policy = res
			
		nodes = (spacenodes..., ts)
		χitp = linear_interpolation(nodes, first.(policy); extrapolation_bc = Flat())
		αitp = linear_interpolation(nodes, last.(policy); extrapolation_bc = Flat())
		Vitp = linear_interpolation(nodes, V; extrapolation_bc = Flat())

		
		jumpresultsmap[ωᵣ] = (χitp, αitp, Vitp, jumpmodels[j])
		
		for Δλ ∈ ΔΛ
			k = findfirst(m -> ωᵣ ≈ m.economy.ωᵣ && Δλ ≈ m.albedo.λ₁ - m.albedo.λ₂, models)

			res = results[k]
			model = models[k]
					
			ts, V, policy = res
				
			nodes = (spacenodes..., ts)
			χitp = linear_interpolation(nodes, first.(policy); extrapolation_bc = Flat())
			αitp = linear_interpolation(nodes, last.(policy); extrapolation_bc = Flat())
			Vitp = linear_interpolation(nodes, V; extrapolation_bc = Flat())

			
			resultsmap[(ωᵣ, Δλ)] = (χitp, αitp, Vitp, model)
		end
	end

    Eᵇ = linear_interpolation(calibration.years .- 2020, calibration.emissions, extrapolation_bc = Line());
end;

function F!(dx, x, p::Tuple{ModelInstance, Any, Any}, t)	
	model, χitp, αitp = p
	
	T, m, y = x
	
	χ = χitp(T, m, y, t)
	α = αitp(T, m, y, t)
	
	dx[1] = μ(T, m, model.hogg, model.albedo) / model.hogg.ϵ
	dx[2] = γ(t, model.economy, model.calibration) - α
	dx[3] = b(t, Point(T, m, y), Policy(χ, α), model)

	return
end;
function F!(dx, x, p::Tuple{ModelBenchmark, Any, Any}, t)	
	model, χitp, αitp = p
	
	T, m, y = x
	
	χ = χitp(T, m, y, t)
	α = αitp(T, m, y, t)
	
	dx[1] = μ(T, m, model.hogg) / model.hogg.ϵ
	dx[2] = γ(t, model.economy, model.calibration) - α
	dx[3] = b(t, Point(T, m, y), Policy(χ, α), model)

	return
end;

function G!(dx, x, p, t)
	model = first(p)
	
	dx[1] = 0. # model.hogg.σₜ / model.hogg.ϵ
	dx[2] = 0.
	dx[3] = 0. # model.economy.σₖ
	
	return
end;

begin # Solver
    ω = first(keys(resultsmap))[1]
	tspan = (0., Economy().t₁)

	problems = OrderedDict{Float64, SDEProblem}()
	fn = SDEFunction(F!, G!)

	for Δλ ∈ ΔΛ
		χitp, αitp, _, model = resultsmap[(ω, Δλ)]
		parameters = (model, χitp, αitp)
		
		problems[Δλ] = SDEProblem(fn, X₀, tspan, parameters)
	end	

    solutions = OrderedDict(key => solve(EnsembleProblem(prob), EnsembleDistributed(); trajectories = 30) for (key, prob) ∈ problems);
end;

begin # Solver for benchmark
	χitp, αitp, _, model = jumpresultsmap[ω]
	parameters = (model, χitp, αitp)
			
	jumpproblem = SDEProblem(fn, X₀, tspan, parameters)
	jumpsolution = solve(EnsembleProblem(jumpproblem), EnsembleDistributed(), trajectories = 30)
end;

begin # Data extraction
    timespan = range(0, 80; step = 0.5)
    function emissionpath(solution, model, αitp)
        E = Matrix{Float64}(undef, length(timespan), length(solution))
    
        for (j, sim) ∈ enumerate(solution), (i, tᵢ) ∈ enumerate(timespan)
            T, m, y = sim(tᵢ)
            αₜ = αitp(T, m, y, tᵢ)
            
            M = exp(m)
            Eₜ = (M / Model.Gtonoverppm) * (γ(tᵢ, model.economy, model.calibration) - αₜ)
    
            
            E[i, j] = Eₜ
        end
    
        return E
    end;
    function consumptionpath(solution, model, χitp)
        C = Matrix{Float64}(undef, length(timespan), length(solution))
    
        for (j, sim) ∈ enumerate(solution), (i, tᵢ) ∈ enumerate(timespan)
            T, m, y = sim(tᵢ)
            χᵢ = χitp(T, m, y, tᵢ)
            
            C[i, j] = exp(y) * χᵢ
        end
    
        return C
    end;
    function variablepath(solution, model)
        X = Matrix{Point}(undef, length(timespan), length(solution))
    
        for (j, sim) ∈ enumerate(solution), (i, tᵢ) ∈ enumerate(timespan)
            T, m, y = sim(tᵢ)
            X[i, j] = Point(T, m, y)
        end
    
        return X
    end;
end

begin # Emission comparison figure
    decadeticks = 0:10:80

    emissionfig = @pgf Axis(
        {
            width = raw"\linewidth",
            height = raw"0.7\linewidth",
            grid = "both",
            xlabel = raw"Year",
            ylabel = raw"Net Emissions",
            xmin = minimum(timespan), xmax = maximum(timespan),
            xtick = decadeticks, xticklabels = decadeticks .+ 2020,
            ymin = -1., ymax = 25.,
        }
    )
    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "skeleton-optemissions.tikz"), emissionfig; include_preamble = true) 
    end

        
    # Albedo
    for Δλ ∈ [0.06, 0.08]
        _, αitp, _, model = resultsmap[(ω, Δλ)];
        solution = solutions[Δλ];
        emissions = emissionpath(solution, model, αitp);

        label = @sprintf("\$\\Delta\\lambda = %.0f \\%%\$", 100 * Δλ)

        for E ∈ eachcol(emissions)
            @pgf push!(
                emissionfig, 
                Plot({forget_plot, color = seqpaletteΔλ[Δλ], opacity = 0.2}, Coordinates(zip(timespan, E))
                )
            )
        end

        @pgf push!(
            emissionfig, 
            Plot({ color = seqpaletteΔλ[Δλ], line_width = "3pt" },
                Coordinates(zip(timespan, emissions))
            ), LegendEntry(label)
        )
    end

    # Jump
    jumpcolor = RGB(0, 77 / 255, 64 / 255)
	_, αitp, _, jumpmodel = jumpresultsmap[ω]
	emissions = emissionpath(jumpsolution, jumpmodel, αitp)

    for E ∈ eachcol(emissions)
        @pgf push!(
            emissionfig, 
            Plot({forget_plot, color = jumpcolor, opacity = 0.2}, Coordinates(zip(timespan, E))
            )
        )
    end

    @pgf push!(
        emissionfig, 
        Plot({ color = jumpcolor, line_width = "3pt" },
            Coordinates(zip(timespan, median(emissions, dims = 2)))
        ), LegendEntry("Stochastic")
    )

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "optemissions.tikz"), emissionfig; include_preamble = true) 
    end


    emissionfig
end

begin # Consumption
    consumptionfig = @pgf Axis(
        {
            width = raw"\linewidth",
            height = raw"0.7\linewidth",
            grid = "both",
            xlabel = raw"Year",
            ylabel = raw"GDP / Consumption [Trillions US \$]",
            no_markers,
            ultra_thick,
            xmin = minimum(timespan), xmax = maximum(timespan),
            xtick = decadeticks, xticklabels = decadeticks .+ 2020,
            ymin = 0., ymax = 200.,
        }
    )

    for Δλ ∈ [0.06, 0.08]
        χitp, _, _, model = resultsmap[(ω, Δλ)];
        solution = solutions[Δλ];

        X = variablepath(solution, model)
        Y = [exp(x.y) for x ∈ X]

        label = @sprintf("\$\\Delta\\lambda = %.0f \\%%\$", 100 * Δλ)

        @pgf push!(
            consumptionfig, 
            Plot({ color = seqpaletteΔλ[Δλ], line_width = "3pt" },
                Coordinates(zip(timespan, median(Y, dims = 2)))
            ), LegendEntry(label)
        )
    end
    
    # Jump
	jumpχitp, _, _, jumpmodel = jumpresultsmap[ω]
	X = variablepath(jumpsolution, jumpmodel)
	Y = [exp(x.y) for x ∈ X]
    

    @pgf push!(
        consumptionfig, 
        Plot({ color = jumpcolor, line_width = "3pt" },
            Coordinates(zip(timespan, median(Y, dims = 2)))
        ), LegendEntry("Stochastic")
    )

    @pgf consumptionfig["legend style"] = raw"at = {(0.9, 0.3)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "optgdp.tikz"), consumptionfig; include_preamble = true) 
    end

    consumptionfig
end

begin # Temperature
    ytick, yticklabels = makedevxlabels(1, 2, first(models); step = 0.25, digits = 2)

    tempfig = @pgf Axis(
        {
            width = raw"\linewidth",
            height = raw"0.7\linewidth",
            grid = "both",
            xlabel = raw"Year",
            ylabel = TEMPLABEL,
            xmin = minimum(timespan), xmax = maximum(timespan),
            xtick = decadeticks, xticklabels = decadeticks .+ 2020, 
            ytick = ytick, yticklabels = yticklabels
        }
    )

    for Δλ ∈ [0.06, 0.08]
        χitp, _, _, model = resultsmap[(ω, Δλ)];
        solution = solutions[Δλ];

        X = variablepath(solution, model)
        temps = [x.T for x ∈ X]

        label = @sprintf("\$\\Delta\\lambda = %.0f \\%%\$", 100 * Δλ)

        @pgf push!(
            tempfig, 
            Plot({ color = seqpaletteΔλ[Δλ], line_width = "3pt" },
                Coordinates(zip(timespan, median(temps, dims = 2)))
            ), LegendEntry(label)
        )

    end
    
    # Jump
	jumpχitp, _, _, jumpmodel = jumpresultsmap[ω]
	X = variablepath(jumpsolution, jumpmodel)
	temps = [x.T for x ∈ X]

    @pgf push!(
        tempfig, 
        Plot({ color = jumpcolor, line_width = "3pt" },
            Coordinates(zip(timespan, median(temps, dims = 2)))
        ), LegendEntry("Stochastic")
    )

    @pgf tempfig["legend style"] = raw"at = {(0.9, 0.3)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "opttemp.tikz"), tempfig; include_preamble = true) 
    end

    tempfig
end

begin # Markov Chain
    model = last(resultsmap[(0.002, 0.08)]);

    y = log(model.economy.Y₀)

    Tdomain = extrema(temperatureticks[1])

    denseG = RegularGrid([Tdomain, domains[2], domains[3]], 101)
    Tdense = range(denseG.domains[1]..., length = size(denseG, 1))
    mdense = range(denseG.domains[2]..., length = size(denseG, 2))
    
    ar = (denseG.domains[2][2] - denseG.domains[2][1]) / (denseG.domains[1][2] - denseG.domains[1][1])

    mnullcline = [mstable(T, model.hogg, model.albedo) for T in Tdense]
    
    tplot = contourf(
        mdense, Tdense, (m, T) -> 1 / (abs(μ(T, m, model.hogg, model.albedo)) + model.hogg.σₜ^2); 
        linewidth = 0, c = :Reds, 
        xlims = denseG.domains[2], ylims = denseG.domains[1],
        title = "Timestep, \$h \\; \\Delta t\$",
        yticks = temperatureticks,
        xlabel = "Log Carbon Concentration, \$m\$",
        ylabel = TEMPLABEL,
        margins = 5Plots.mm, aspect_ratio = ar, dpi = 180
    )

    # plot!(tplot, mnullcline, Tdense; c = :white, label = false, linewidth = 2, linestyle = :dash)

    SAVEFIG && savefig(tplot, joinpath(PLOTPATH, "timestep.png"))

    tplot
end

begin # Abatement policy
    _, αitp, _, model = resultsmap[(0.002, 0.08)];

    mnullcline = [mstable(T, model.hogg, model.albedo) for T in Tdense]

    
    αplot = contourf(
        mdense, Tdense, (m, T) -> αitp(T, m, y, 0.); 
        linewidth = 0, c = :BuGn_3, 
        xlims = denseG.domains[2], ylims = denseG.domains[1],
        title = "Abatement, \$ \\alpha(m, T, Y) \$",
        yticks = temperatureticks,
        xlabel = "Log Carbon Concentration, \$m\$",
        ylabel = TEMPLABEL,
        margins = 5Plots.mm, aspect_ratio = ar, dpi = 180,
        levels = 100
    )

    plot!(αplot, mnullcline, Tdense; c = :black, label = false, linewidth = 2)

    SAVEFIG && savefig(αplot, joinpath(PLOTPATH, "abatement.png"))
   
    αplot
end