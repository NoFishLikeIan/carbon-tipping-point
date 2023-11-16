using Revise
using JLD2, DotEnv, CSV
using DataFrames

using UnPack
using Dierckx
using LinearAlgebra

using ChangePrecision

using DifferentialEquations, DifferentialEquations.EnsembleAnalysis

using KernelDensity
using Plots, Printf, PGFPlotsX, Colors
using Model, Utils

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

    SAVEFIG = true 
end

economy = Model.Economy();
hogg = Model.Hogg();
albedo = Model.Albedo();

instance = (economy, hogg, albedo);
calibration = load(joinpath(DATAPATH, "calibration.jld2"), "calibration");

secondtoyears = 3.154f7
mass = diagm([hogg.ϵ / secondtoyears, 1.])

kelvintocelsius = 273.15
xpreindustrial = 14 + kelvintocelsius

function stringtempdev(x::Real; digits = 2)
    fsign = x > 0 ? "+" : ""
    fmt = Printf.Format("$fsign%0.$(digits)f")
    return Printf.format(fmt, x)
end

function makedevxlabels(from, to, climate; step = 0.5, withcurrent = false, digits = 2)

    preindustrialx = range(from, to; step = step)
    xticks = preindustrialx .+ xpreindustrial

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

function extractoptimalemissions(σₓ, sim, e::Function; Tsim = 1001)
	T = first(sim).t |> last
	timespan = range(0, T; length = Tsim)
	nsim = size(sim, 3)

	optemissions = Matrix{Float64}(undef, Tsim, nsim)

	for idxsim in 1:nsim
		optemissions[:, idxsim] .= [e(x, m, σₓ) for (x, m) in sim[idxsim](timespan).u]
	end

	return optemissions
end

begin # labels and axis
    TEMPLABEL = raw"Temperature deviations $T - T^{\mathrm{p}}$"
    Tspacedev = range(0f0, 10f0; length = 51)
    Tspace = Tspacedev .+ hogg.Tᵖ
    yearlytime = collect(0:economy.t₁) 
    ΔTᵤ = last(Tspace) - first(Tspace)
    temperatureticks = makedevxlabels(0., ΔTᵤ, (hogg, albedo); step = 1, digits = 0)  
end

@changeprecision Float32 begin # Load IPCC data
    # Import IPCC data
    IPCCDATAPATH = joinpath(DATAPATH, "climate-data", "proj-median.csv")
    ipccproj = CSV.read(IPCCDATAPATH, DataFrame)

    getscenario(s::Int64) = filter(:Scenario => isequal("SSP$(s) - Baseline"), ipccproj)

    bauscenario = getscenario(5)
end

begin # Albedo plot
    
    Δλmap = [0.02, 0.06, 0.08] 
    seqpaletteΔλ = generateseqpalette(length(Δλmap))
    
    albedovariation = [(T -> Model.λ(T, Albedo(λ₂ = albedo.λ₁ - Δλ))).(Tspace) for Δλ ∈ Δλmap]


    albedofig = @pgf Axis(
        {
            width = raw"1\textwidth",
            height = raw"0.6\textwidth",
            grid = "both",
            xlabel = TEMPLABEL,
            ylabel = raw"Albedo coefficient $\lambda(T)$",
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
    nullclinevariation = [(T -> Model.Mstable(T, hogg, Albedo(λ₂ = albedo.λ₁ - Δλ))).(Tspace) for Δλ ∈ Δλmap]


    nullclinefig = @pgf Axis(
        {
            width = raw"0.7\textwidth",
            height = raw"0.7\textwidth",
            grid = "both",
            ylabel = TEMPLABEL,
            xlabel = raw"Carbon concentration $M$",
            xmin = hogg.Mᵖ, xmax = 1200,
            xtick = 200:200:1200,
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
    
    gdata = [Model.γ(t, economy, calibration) for t ∈ yearlytime]
    coords = Coordinates(zip(yearlytime, gdata))

    markers = @pgf Plot({ only_marks, mark_options = {fill = gcolor, scale = 1.5, draw_opacity = 0}, mark_repeat = 10}, coords) 

    curve = @pgf Plot({color = gcolor}, coords) 

    push!(gfig, curve, markers)

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "growthmfig.tikz"), gfig; include_preamble = true) 
    end

    gfig
end

"Drift dynamics of (T, m)"
function Fbau!(du, u, parameters, t)
	instance, calibration = parameters
    economy, hogg, albedo = instance

	du[1] = Model.μ(u[1], u[2], hogg, albedo)
	du[2] = Model.γ(t, economy, calibration)
end
function G!(du, u, parameters, t)
	instance = first(parameters)
    hogg = instance[2]

	du[1] = hogg.σ²ₜ 
	du[2] = 0f0
end

function Fterm!(du, u, parameters, t)
	instance, calibration = parameters
    economy, hogg, albedo = instance

	du[1] = Model.μ(u[1], u[2], hogg, albedo)
	du[2] = 0f0
end

function simulatebau(Δλ::Float32; trajectories = 1000, tspan = (0f0, last(yearlytime))) # Business as Usual, ensemble simulation    
    baualbedo = Albedo(λ₂ = albedo.λ₁ - Δλ)    
    bauparameters = ((economy, hogg, baualbedo), calibration)
    
    fn = SDEFunction(Fbau!, G!, mass_matrix = mass)
    
    problembse = SDEProblem(fn, G!, [hogg.T₀, log(hogg.M₀)], tspan, bauparameters)
    
    ensemblebse = EnsembleProblem(problembse)
    
    bausim = solve(ensemblebse, trajectories = trajectories)
    baunullcline = (x -> Model.mstable(x, hogg, baualbedo)).(Tspace)
    
    return bausim, baunullcline
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
            xmin = hogg.Mᵖ, xmax = 1200,
            xtick = 200:100:1100,
            grid = "both"
        }
    )

    @pgf for (i, Δλ) ∈ enumerate([0f0, 8f-2])
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
        baumedianT = @. first(baumedian) - hogg.Tᵖ


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
            xpath = @. first(path) - hogg.Tᵖ

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

begin # Density of business as usual scenario
    yearsofdensity = 10:10:80
    densedomain = collect(0:0.1f0:12)
    
    baupossim, _ = simulatebau(8f-2; trajectories = 2001)
    decadetemperatures = [first(componentwise_vectors_timepoint(baupossim, t)) .- hogg.Tᵖ for t in yearsofdensity]
    dists = (x -> kde(x)).(decadetemperatures)
    densities = [x -> pdf(d, x) for d in dists]

    poscolor = PALETTE[1]
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
            ytick = collect(yearsofdensity),
            x_dir = "reverse",
            xlabel = raw"Temperature deviations $T - T^{\mathrm{p}}$",
            ylabel = raw"Year",
            zlabel = raw"Density of temperature",
            yticklabels = yearsofdensity .+ BASELINE_YEAR,
            tick_label_style = {scale = 0.5}
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
        PGFPlotsX.save(joinpath(PRESENTATIONPATH, "bau-x-dens.tikz"), densityfig; include_preamble = true) 
    end

    densityfig
end

begin # Carbon decay calibration
    sinkspace = range(hogg.N₀, 1.2f0 * hogg.N₀; length = 101)
    decay = [Model.δₘ(N, hogg) for N in sinkspace]

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
    bausim, baunullcline = simulatebau(0f0; trajectories = 1)
    M = exp.([u[2] for u in bausim[1].u])
    decaysim = Model.δₘ.(M, Ref(hogg))
    
    Msparse = exp.([Float32(bausim(y)[1][2]) for y in 0:10:horizon])
    decaysimsparse = Model.δₘ.(Msparse, Ref(hogg))

    decaypathfig = @pgf Axis(
        {
            width = raw"0.7\textwidth",
            height = raw"0.5\textwidth",
            grid = "both",
            xlabel = raw"Carbon concentration $M$",
            ylabel = raw"Decay of CO$_2$ in the atmosphere $\delta_m$",
            no_markers,
            ultra_thick,
            xmin = hogg.M₀, xmax = maximum(M),
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


begin # Albedo plot
    damagefig = @pgf Axis(
        {
            width = raw"1\textwidth",
            height = raw"0.6\textwidth",
            grid = "both",
            xlabel = TEMPLABEL,
            ylabel = raw"Depreciation rate of capital $\delta_k(T)$",
            xticklabels = temperatureticks[2],
            xtick = 0:1:ΔTᵤ,
            no_markers,
            ultra_thick,
            xmin = 0, xmax = ΔTᵤ,
            ytick = 0:0.1:1
        }
    )

    @pgf damagecurve = Plot({color = PALETTE[2]},
        Coordinates(Tspacedev, [Model.d(T, economy, hogg) for T in Tspace])
    )

    push!(damagefig, damagecurve)

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "damagefig.tikz"), damagefig; include_preamble = true) 
    end

    damagefig
end

begin
    χspace = range(0f0, 1f0; length = 21)

    outputfig = @pgf Axis(
        {
            width = raw"\linewidth",
            height = raw"\linewidth",
            grid = "both",
            ylabel = raw"$\phi(t, 1 - \chi)$",
            xlabel = raw"$\chi$",
            xtick = 0:0.2:1,
            ytick = 0:0.1:0.2,
            xmin = 0, xmax = 1,
            ymin = 0, ymax = 0.2,
            ultra_thick, xticklabel_style = {rotate = 45},
            legend_style={nodes={scale = 0.5}},
        }
    )

    for (i, t) ∈ enumerate([0f0, 50f0])
        data = [Model.ϕ(t, χ, economy) for χ ∈ χspace]
        curve = @pgf Plot({color = PALETTE[i]}, Coordinates(zip(χspace, data))) 

        push!(outputfig, curve, LegendEntry("\$t = $t\$"))
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "phifig.tikz"), outputfig; include_preamble = true) 
    end

    outputfig
end
