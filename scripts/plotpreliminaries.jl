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
include("utils/simulating.jl")

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
    jumpcolor = RGB(0, 77 / 255, 64 / 255)

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
    presentationtemperatureticks = makedevxlabels(0., ΔTᵤ, first(models); step = 2, digits = 0)
end

begin # Load IPCC data
    IPCCDATAPATH = joinpath(DATAPATH, "climate-data", "proj-median.csv")
    ipccproj = CSV.read(IPCCDATAPATH, DataFrame)

    getscenario(s::Int64) = filter(:Scenario => isequal("SSP$(s) - Baseline"), ipccproj)

    bauscenario = getscenario(5)
    seqpaletteΔλ = Dict(ΔΛ .=> generateseqpalette(length(ΔΛ)))
end

begin # Albedo plot
    ΔΛfig = [0.06, 0.08]
    albedovariation = [(T -> Model.λ(T, Albedo(λ₂ = Albedo().λ₁ - Δλ))).(Tspace) for Δλ ∈ ΔΛfig]


    albedofig = @pgf Axis(
        {
            width = raw"0.4\textwidth",
            height = raw"0.4\textwidth",
            grid = "both",
            xlabel = TEMPLABEL,
            ylabel = raw"$\lambda(T)$",
            xticklabels = presentationtemperatureticks[2],
            xtick = 0:2:ΔTᵤ,
            no_markers, ultra_thick,
            xmin = 0, xmax = (ΔTᵤ),
            ymin = 0.2, ymax = 0.32,
            ytick = 0.2:0.05:0.35
        }
    )

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "skeleton-albedo.tikz"), albedofig; include_preamble = true) 
    end

    @pgf for (Δλ, albedodata) in zip(ΔΛfig, albedovariation)
        curve = Plot(
            {color=seqpaletteΔλ[Δλ], line_width="0.4em"}, 
            Coordinates(
                collect(zip(Tspacedev, albedodata))
            )
        ) 

        label = @sprintf("%.0f \\%%", 100 * Δλ)
        legend = LegendEntry(label)

        push!(albedofig, curve) # legend)
    end
    @pgf albedofig["legend style"] = raw"at = {(0.3, 0.5)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "poster_albedo.tikz"), albedofig; include_preamble = true) 
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
    
    gdata = [γ(t, first(models).calibration) for t ∈ yearlytime]
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
            yticklabels = presentationtemperatureticks[2],
            ytick = 0:2:ΔTᵤ,
            ymin = 0, ymax = ΔTᵤ,
            xmin = Hogg().Mᵖ, xmax = 1200,
            xtick = 200:100:1100,
            grid = "both"
        }
    )

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "baufig.tikz"), baufig; include_preamble = true) 
    end    

    @pgf for (i, Δλ) ∈ enumerate([0.08])
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

begin # Side by side Jump and Λ
    baufig = @pgf GroupPlot(
        {
            group_style = { 
                group_size = "2 by 1", 
                horizontal_sep="0pt",
                yticklabels_at="edge left"
            }, 
            width = raw"0.4\textwidth",
            height = raw"0.3\textwidth",
            yticklabels = presentationtemperatureticks[2],
            ytick = 0:2:ΔTᵤ,
            ymin = 0, ymax = ΔTᵤ,
            xmin = Hogg().Mᵖ, xmax = 1200,
            xtick = 200:200:1100,
            grid = "both"
        }
    )

    # Jump simulation
    bausim, baunullcline = simulatebaujump(trajectories = 10);
    baumedian = timeseries_point_median(bausim, yearlytime)
    baumedianM = @. exp(last(baumedian.u))
    baumedianT = @. first(baumedian.u) - Hogg().Tᵖ

    jumpfigs = []

    # Nullcline
    push!(jumpfigs, @pgf Plot({dashed, color = "black", ultra_thick, forget_plot}, Coordinates(zip(exp.(baunullcline), Tspacedev))))

    mediancoords = Coordinates(zip(baumedianM, baumedianT))

    @pgf push!(
        jumpfigs,
        Plot({ line_width="0.3em", color = jumpcolor }, mediancoords),
        LegendEntry("Stochastic"),
        Plot({ only_marks, mark_options = { fill = jumpcolor, scale = 1.5, draw_opacity = 0 }, mark_repeat = 20, forget_plot, mark = "*"}, mediancoords)
    )

    @pgf for sim in bausim
        path = sim.(yearlytime)

        mpath = @. exp([u[2] for u in path])
        xpath = @. first(path) - Hogg().Tᵖ

        push!(
            jumpfigs, 
            Plot({forget_plot, color = jumpcolor, opacity = 1},
                Coordinates(zip(mpath, xpath)),
            )
        )
    end

    @pgf push!(baufig,  { xlabel = raw"Carbon concentration $M$", ylabel = TEMPLABEL }, jumpfigs...)

    
    # Data simulation
    Δλfigs = []

    for Δλ ∈ [0.06, 0.08]
        bausim, baunullcline = simulatebau(Δλ; trajectories = 10)
        baumedian = [timepoint_median(bausim, t) for t in yearlytime]
        baumedianM = @. exp([u[2] for u in baumedian])
        baumedianT = @. first(baumedian) - Hogg().Tᵖ

        timeseriescolor = seqpaletteΔλ[Δλ]

        # Nullcline
        push!(Δλfigs,
            @pgf Plot({dashed, color = "black", ultra_thick, forget_plot},
                Coordinates(zip(exp.(baunullcline), Tspacedev))
            )
        )

        mediancoords = Coordinates(zip(baumedianM, baumedianT))

        label = @sprintf "%.0f\\%% loss" Δλ * 100

        @pgf begin
            push!(
                Δλfigs,
                Plot({line_width="0.3em", color = timeseriescolor, opacity = 0.8}, mediancoords),
                LegendEntry(label),
                Plot({only_marks, mark_options = {fill = timeseriescolor, scale = 1.5, draw_opacity = 0, fill_opacity = 0.8}, mark_repeat = 20, forget_plot, mark = "*"}, mediancoords)
            )
        end

        @pgf for sim in bausim
            path = sim.(yearlytime)

            mpath = @. exp([u[2] for u in path])
            xpath = @. first(path) - Hogg().Tᵖ

            push!(
                Δλfigs, 
                Plot({forget_plot, color = timeseriescolor, opacity = 0.2},
                    Coordinates(zip(mpath, xpath)),
                )
            )
        end
    end

    @pgf push!(baufig,  { xlabel = raw"Carbon concentration $M$" }, Δλfigs...)
  
    @pgf baufig["legend style"] = raw"at = {(0.45, 0.95)}"


    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "poster_baufig.tikz"), baufig; include_preamble = true) 
    end
    
    baufig
end

begin # Individual BAU plots
    Δλ = 0.08
    baufig = @pgf Axis(
        {
            width = raw"\linewidth",
            height = raw"\linewidth",
            ylabel = "Temperature",
            xlabel = raw"CO$_2$",
            yticklabels = temperatureticks[2][1:2:end],
            ytick = 0:2:ΔTᵤ,
            ymin = 0, ymax = ΔTᵤ,
            xmin = Hogg().Mᵖ, xmax = 1200,
            xtick = 200:300:1300,
            grid = "both"
        }
    )

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "introskeleton-baufig.tikz"), baufig; include_preamble = true) 
    end
    
    # IPCC benchmark line
    ipcccoords = Coordinates(zip(bauscenario[3:end, "CO2 concentration"], bauscenario[3:end, "Temperature"]))

    ipccbau = @pgf Plot({
        very_thick, 
        color = "black", 
        mark = "*", 
        mark_options = {scale = 1.5, draw_opacity = 0}, 
        mark_repeat = 2
    }, ipcccoords)

    push!(
        baufig, 
        # LegendEntry("SSP5 - Baseline"), 
        ipccbau
    )

    # Data simulation
    bausim, baunullcline = simulatebau(Δλ; trajectories = 10);
    baumedian = timeseries_point_median(bausim, yearlytime)
    baumedianM = @. exp(last(baumedian.u))
    baumedianT = @. first(baumedian.u) - Hogg().Tᵖ

    # Nullcline
    push!(baufig, # LegendEntry(raw"$\mu(T, M) = 0$"),
        @pgf Plot({dashed, color = "black", ultra_thick},
            Coordinates(collect(zip(exp.(baunullcline), Tspacedev))))
    )

    mediancoords = Coordinates(zip(baumedianM, baumedianT))

    label = @sprintf("\$\\Delta\\lambda = %.0f \\%%\$", 100 * Δλ)

    @pgf push!(
        baufig,
        Plot({ line_width="0.05cm", color = seqpaletteΔλ[Δλ] }, mediancoords),
        # LegendEntry(label),
        Plot({ only_marks, mark_options = { fill = seqpaletteΔλ[Δλ], scale = 1.5, draw_opacity = 0 }, mark_repeat = 20, forget_plot, mark = "*"}, mediancoords)
    )

    @pgf for sim in bausim
        path = sim.(yearlytime)

        mpath = @. exp([u[2] for u in path])
        xpath = @. first(path) - Hogg().Tᵖ

        push!(
            baufig, 
            Plot({forget_plot, color = seqpaletteΔλ[Δλ], opacity = 1.},
                Coordinates(zip(mpath, xpath)),
            )
        )
    end

    @pgf baufig["legend style"] = raw"at = {(0.4, 0.975)}"


    if SAVEFIG
        filelabel = @sprintf("%.0f", 100 * Δλ)
        PGFPlotsX.save(joinpath(PLOTPATH, "intro_baufig_$filelabel.tikz"), baufig; include_preamble = true) 
    end

    baufig
end

begin # Jump process
    baufig = @pgf Axis({
            width = raw"\linewidth",
            height = raw"\linewidth",
            ylabel = "Temperature",
            xlabel = raw"CO$_2$",
            yticklabels = temperatureticks[2][1:2:end],
            ytick = 0:2:ΔTᵤ,
            ymin = 0, ymax = ΔTᵤ,
            xmin = Hogg().Mᵖ, xmax = 1200,
            xtick = 200:300:1300,
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

    push!(
        baufig, 
        # LegendEntry("SSP5 - Baseline"), 
        ipccbau
    )

    # Data simulation
    bausim, baunullcline = simulatebaujump(trajectories = 10);
    baumedian = timeseries_point_median(bausim, yearlytime)
    baumedianM = @. exp(last(baumedian.u))
    baumedianT = @. first(baumedian.u) - Hogg().Tᵖ

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "skeleton-baufig.tikz"), baufig; include_preamble = true) 
    end

    # Nullcline
    push!(baufig, 
        # LegendEntry(raw"$\mu(T, M) = 0$"),
        @pgf Plot({dashed, color = "black", ultra_thick},
            Coordinates(collect(zip(exp.(baunullcline), Tspacedev))))
    )

    mediancoords = Coordinates(zip(baumedianM, baumedianT))

    label = "Stochastic"


    @pgf push!(
        baufig,
        Plot({ line_width="0.1cm", color = jumpcolor }, mediancoords),
        # LegendEntry(label),
        Plot({ only_marks, mark_options = { fill = jumpcolor, scale = 1.5, draw_opacity = 0 }, mark_repeat = 20, forget_plot, mark = "*"}, mediancoords)
    )

    @pgf for sim in bausim
        path = sim.(yearlytime)

        mpath = @. exp([u[2] for u in path])
        xpath = @. first(path) - Hogg().Tᵖ

        push!(
            baufig, 
            Plot({forget_plot, color = jumpcolor, opacity = 1},
                Coordinates(zip(mpath, xpath)),
            )
        )
    end

    @pgf baufig["legend style"] = raw"at = {(0.4, 0.975)}"


    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "intro_baufig_jump.tikz"), baufig; include_preamble = true) 
    end

    baufig
end

begin # Jump illustration
    η = [intensity(T, Hogg(), Jump()) for T in Tspace];
    ϵ = [increase(T, Hogg(), Jump()) for T in Tspace];

    illjumpfig = @pgf TikzPicture()

    ηaxis = @pgf Axis(
        {
            grid = "both",
            width = raw"0.4\textwidth",
            height = raw"0.4\textwidth",
            xlabel = TEMPLABEL,
            xticklabels = presentationtemperatureticks[2],
            xtick = 0:2:ΔTᵤ,
            ytick = 0:0.2:0.8,
            no_markers, ultra_thick,
            xmin = 0, xmax = ΔTᵤ,
            ylabel_style = { color = PALETTE[1] },
            "axis y line*" = "left",
            ylabel = "\$\\eta\$",
            ymin = 0, ymax = maximum(η) * 1.05
        }
    ); push!(illjumpfig, ηaxis);

    ηcurve = @pgf Plot({ line_width="0.4em", color = PALETTE[1] }, Coordinates(zip(Tspacedev, η))); 
    
    push!(ηaxis, ηcurve)

    ϵaxis = @pgf Axis(
        {
            no_markers, ultra_thick,
            width = raw"0.4\textwidth",
            height = raw"0.4\textwidth",
            "axis y line*" = "right",
            ytick = 0:0.1:0.3,
            "hide x axis",
            xmin = 0, xmax = ΔTᵤ,
            axis_x_line = "none",
            ylabel_style = {color = generateseqpalette(4)[1] },
            ylabel = "\$\\epsilon\$",
            ymin = 0, ymax = maximum(ϵ) * 1.05,
        }
    ); push!(illjumpfig, ϵaxis);

    ϵcurve = @pgf Plot({ line_width="0.4em", color = generateseqpalette(4)[1]  }, Coordinates(zip(Tspacedev, ϵ))); 
    
    push!(ϵaxis, ϵcurve)

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "illjumpfig.tikz"), illjumpfig; include_preamble = true) 
    end

    illjumpfig
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
    bausim, baunullcline = simulatebau(0.06; trajectories = 1)
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
