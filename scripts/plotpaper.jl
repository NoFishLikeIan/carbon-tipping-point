using JLD2, DotEnv

using UnPack
using Dierckx
using LinearAlgebra

using DifferentialEquations, DifferentialEquations.EnsembleAnalysis

using KernelDensity
using Plots, Printf, PGFPlotsX, Colors

begin # Global variables
    env = DotEnv.config()

    PALETTE = color.(["#003366", "#E31B23", "#005CAB", "#DCEEF3", "#FFC325", "#E6F1EE"])
    SEQPALETTECODE = :YlOrRd
    generateseqpalette(n) = palette(SEQPALETTECODE, n + 2)[3:end]

    LINESTYLE = ["solid", "dashed", "dotted"]
    
    BASELINE_YEAR = parse(Int64, get(env, "BASELINE_YEAR", "2020"))
    DATAPATH = get(env, "DATAPATH", "data")
    PLOTPATH = get(env, "PLOTPATH", "plots")

    SAVEFIG = false 
end

include("../src/model/climate.jl")
include("../src/model/economy.jl")

include("../src/utils/plotting.jl")
include("../src/utils/dynamics.jl")

begin # Initialise models and set domains
    albedo = Albedo()
    baseline = Hogg(σ²ₜ = 1f-1)
    climate = (baseline, albedo)

    Tₗ, Tᵤ = baseline.Tᵖ, baseline.Tᵖ + 13
    
    Mₗ, Mᵤ = baseline.Mᵖ, mstable(Tₗ, climate)
    
    Tspace = range(Tₗ, Tᵤ; length = 201)
    Mspace = range(Mₗ, Mᵤ; length = 201)
    Tspacedev = collect(Tspace .- baseline.Tᵖ)
    
    nullclinecarbon = (x -> mstable(x, climate)).(Tspace)

    economy = Economy()
end

begin # Load calibrated data
    @load joinpath(DATAPATH, "calibration.jld2") ipcc
    @unpack Eᵇ, Tᵇ, Mᵇ, N₀, γparameters = ipcc

    γᵇ(t) = γ(t, Float32.(γparameters[1:3]), Float32.(γparameters[4]))
end

# -- Climate dynamics plots
TEMPLABEL = raw"Temperature deviations $T - T^{\mathrm{p}}$"

begin # Albedo plot

    ΔTᵤ = last(Tspace) - baseline.Tᵖ
    
    Δλmap = [0.02, 0.06, 0.08] 
    seqpaletteΔλ = generateseqpalette(length(Δλmap))
    
    temperatureticks = makedevxlabels(0., ΔTᵤ, climate; step = 1, digits = 0)
    albedovariation = [(T -> λ(T, Albedo(λ₂ = albedo.λ₁ - Δλ))).(Tspace) for Δλ ∈ Δλmap]


    albedofig = @pgf Axis(
        {
            width = raw"1\textwidth",
            height = raw"0.6\textwidth",
            grid = "both",
            xlabel = TEMPLABEL,
            ylabel = raw"Albedo coefficient $\lambda(x)$",
            xticklabels = temperatureticks[2],
            xtick = 0:1:ΔTᵤ,
            no_markers,
            ultra_thick,
            xmin = 0, xmax = ΔTᵤ
        }
    )

    @pgf for (i, albedodata) in enumerate(albedovariation)
        curve = Plot(
            {color=seqpaletteΔλ[i], ultra_thick}, 
            Coordinates(
                collect(zip(Tspacedev, albedodata))
            )
        ) 

        legend = LegendEntry("$(Δλmap[i])")

        push!(albedofig, curve, legend)
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "albedo.tikz"), albedofig; include_preamble = true) 
    end

    albedofig
end

begin # Nullcline plot
    nullclinevariation = [(T -> Mstable(T, (baseline, Albedo(λ₂ = albedo.λ₁ - Δλ)))).(Tspace) for Δλ ∈ Δλmap]


    nullclinefig = @pgf Axis(
        {
            width = raw"0.7\textwidth",
            height = raw"0.7\textwidth",
            grid = "both",
            ylabel = TEMPLABEL,
            xlabel = raw"Carbon concentration $M$",
            xmax = 1200,
            xtick = 0:300:1200,
            yticklabels = temperatureticks[2],
            ytick = 0:1:ΔTᵤ,
            ymin = 0, ymax = ΔTᵤ,
            ultra_thick, 
        }
    )

    @pgf for (i, nullclinedata) in enumerate(nullclinevariation)
        coords = Coordinates(collect(zip(nullclinedata, Tspacedev)))

        curve = Plot({color = seqpaletteΔλ[i]}, coords) 

        legend = LegendEntry("$(Δλmap[i])")

        push!(nullclinefig, curve, legend)
    end

    @pgf nullclinefig["legend style"] = raw"at = {(0.3, 0.95)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "nullcline.tikz"), nullclinefig; include_preamble = true) 
    end

    nullclinefig
end

function simulatebau(Δλ; trajectories = 1000) # Business as Usual, ensemble simulation    
    αbau = (T, M) -> 0.
    baualbedo = Albedo(λ₂ = albedo.λ₁ - Δλ)
    
    bauparameters = ((Hogg(), baualbedo), γᵇ, αbau)
    
    SDEFunction(F!, G!, mass_matrix = mass_matrix(baseline))
    
    problembse = SDEProblem(SDEFunction(F!, G!, mass_matrix = mass_matrix(baseline)), G!, [baseline.T₀, log(baseline.M₀), N₀], (0, T), bauparameters)
    
    ensemblebse = EnsembleProblem(problembse)
    
    bausim = solve(ensemblebse, trajectories = trajectories)
    baunullcline = (x -> mstable(x, (baseline, baualbedo))).(Tspace)
    
    return bausim, baunullcline
end

T = 80
yearlytime = 0:1:T

begin # Growth of carbon concentration
    gcolor = first(PALETTE)

    gfig = @pgf Axis(
        {
            width = raw"0.75\linewidth",
            height = raw"0.75\linewidth",
            grid = "both",
            ylabel = raw"Growth rate $\gamma^{\mathrm{b}}$",
            xlabel = raw"Year",
            xtick = 0:20:T,
            xmin = 0, xmax = T,
            xticklabels = BASELINE_YEAR .+ (0:20:T),
            ultra_thick, xticklabel_style = {rotate = 45}
        }
    )   
    
    gdata = γᵇ.(yearlytime)
    coords = Coordinates(zip(yearlytime, gdata))

    markers = @pgf Plot({ only_marks, mark_options = {fill = gcolor, scale = 1.5, draw_opacity = 0}, mark_repeat = 10}, coords) 

    curve = @pgf Plot({color = gcolor}, coords) 

    push!(gfig, curve, markers)

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "growthmfig.tikz"), gfig; include_preamble = true) 
    end

    gfig
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
            xmin = baseline.Mᵖ, xmax = 1200,
            xtick = 200:100:1100,
            grid = "both"
        }
    )

    @pgf for (i, Δλ) ∈ enumerate([0, 0.08])
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
            Coordinates(zip(Mᵇ[3:end], Tᵇ[3:end]))
        )

        push!(Δλplots, ipccbau)
        
        push!(Δλplots, LegendEntry("SSP5 - Baseline"))


        # Data simulation
        bausim, baunullcline = simulatebau(Δλ; trajectories = 20)
        baumedian = [timepoint_median(bausim, t) for t in yearlytime]
        baumedianM = @. exp([u[2] for u in baumedian])
        baumedianT = @. first(baumedian) - baseline.Tᵖ


        # Nullcline
        push!(Δλplots,
            @pgf Plot({dashed, color = "black", ultra_thick, forget_plot},
                Coordinates(collect(zip(exp.(baunullcline), Tspacedev)))
            )
        )

        mediancoords = Coordinates(zip(baumedianM, baumedianT))

        label = isfirst ? raw"$\Delta \lambda = 0.02$" : raw"$\Delta \lambda = 0.08$"

        @pgf begin
            push!(
                Δλplots,
                Plot({ultra_thick, color = timeseriescolor, opacity = 0.8},mediancoords),
                LegendEntry(label),
                Plot({only_marks, mark_options = {fill = timeseriescolor, scale = 1.5, draw_opacity = 0, fill_opacity = 0.8}, mark_repeat = 20, forget_plot, mark = "*"}, mediancoords)
            )
        end

        @pgf for sim in bausim
            path = sim.(yearlytime)

            mpath = @. exp([u[2] for u in path])
            xpath = @. first(path) - baseline.Tᵖ

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

yearsofdensity = 10:10:80
densedomain = collect(0:0.1:12)

baupossim, _ = simulatebau(0.08; trajectories = 51)
decadetemperatures = [first(componentwise_vectors_timepoint(baupossim, t)) .- baseline.Tᵖ for t in yearsofdensity]
dists = (x -> kde(x)).(decadetemperatures)
densities = [x -> pdf(d, x) for d in dists]

begin # Density of business as usual scenario
    poscolor = PALETTE[2]
    densedomain_ext = [[densedomain[1]]; densedomain; [densedomain[end]]]

    densityfig = @pgf Axis(
        {
            width = raw"1\textwidth",
            height = raw"0.6\textwidth",
            grid = "both",
            xmax = densedomain[end], xmin = densedomain[1], 
            zmin = 0, 
            ymin = 0, ymax = 90,
            set_layers,
            view = "{-28}{50}",   # viewpoint
            ztick = collect(0:0.25:1.5),
            ytick = collect(yearsofdensity),
            x_dir = "reverse",
            xlabel = raw"Temperature deviations $x_t - x^{\mathrm{p}}$",
            ylabel = raw"Year",
            zlabel = raw"Density of temperature",
            yticklabels = yearsofdensity .+ BASELINE_YEAR
        },
    )

    @pgf for i in eachindex(dists)
    
        # add a pdf-curve on top of each second data set
        curve = Plot3(
            {
                no_marks,
                style = {thick},
                color = poscolor
            },
            Table(
                x = densedomain,
                y = 10i * ones(length(densedomain)),
                z = densities[i].(densedomain)
            )
        )

        fill = Plot3(
            {
                draw = "none",
                fill = poscolor,
                fill_opacity = 0.25
            },
            Table(x = densedomain_ext,
                    y = 10i * ones(length(densedomain_ext)),
                    z = [[0]; densities[i].(densedomain); [0]])
        )
        push!(densityfig, curve, fill)
    end 

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "bau-x-dens.tikz"), densityfig; include_preamble = true) 
    end

    densityfig
end