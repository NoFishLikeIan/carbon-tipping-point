using UnPack

using Dierckx
using LinearAlgebra

using DifferentialEquations
using DifferentialEquations.EnsembleAnalysis
using DiffEqParamEstim, Optimization, OptimizationOptimJL

using KernelDensity

using CSV, DataFrames, JLD2

using Plots, Printf, PGFPlotsX, Colors

begin # Global variables
    PALETTE = color.(["#003366", "#E31B23", "#005CAB", "#DCEEF3", "#FFC325", "#E6F1EE"])
    SEQPALETTECODE = :YlOrRd
    generateseqpalette(n) = palette(SEQPALETTECODE, n + 2)[3:end]

    LINESTYLE = ["solid", "dashed", "dotted"]
    
    BASELINE_YEAR = 2020 # Year for data

    PLOTPATH = "plots"
    DATAPATH = "data/climate-data"
    SAVEFIG = true 
end

include("../src/model/climate.jl")
include("../src/model/economic.jl")

include("../src/utils/plotting.jl")
include("../src/utils/dynamics.jl")

# -- Load data
function gettimestamp(filepath)
    filename = replace(filepath, ".jld2" => "")
    timestamp = eachsplit(filename, "_") |> collect |> last
    inttimestamp = parse(Int, timestamp)

    return inttimestamp
end

if false # Load the latest simulation
    datapath = "data/sims"; @assert ispath(datapath)
    availablesims = map(gettimestamp, readdir(datapath))

    data = load(joinpath(datapath, "valuefunction_$(maximum(availablesims)).jld2"))
    parameters = get(data, "parameters",  [])
    simulationresults = get(data, "solution",  [])
    @unpack Γ = first(simulationresults)
end

begin # This assumes that all simulations have the same limits in (x, c)

    albedo = Albedo()
    baseline = Hogg(σ²ₜ = 0.1)
    climate = (baseline, albedo)

    Tₗ, Tᵤ = baseline.Tᵖ, baseline.Tᵖ + 13
    
    Mₗ, Mᵤ = baseline.Mᵖ, mstable(Tₗ, climate)
    
    Tspace = range(Tₗ, Tᵤ; length = 201)
    Mspace = range(Mₗ, Mᵤ; length = 201)
    Tspacedev = collect(Tspace .- baseline.Tᵖ)
    
    nullclinecarbon = (x -> mstable(x, climate)).(Tspace)

    economy = Economy()
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

begin # Import IPCC data
    ipccdatapath = joinpath(DATAPATH, "proj-median.csv")
    ipccproj = CSV.read(ipccdatapath, DataFrame)
    
    getscenario(s::Int64) = filter(:Scenario => isequal("SSP$(s) - Baseline"), ipccproj)

    bauscenario = getscenario(5)
    ipcctime = bauscenario.Year .- BASELINE_YEAR
    T = last(ipcctime)

    Mᵇ = bauscenario[:, "CO2 concentration"]
    Tᵇ = bauscenario[:, "Temperature"]
    Eᵇ = (Gtonoverppm / 1e9) * bauscenario[:, "CO2 emissions"]

    # Calibrate g
    gcalib_data = Array(log.(Mᵇ)')
    t0 = first(ipcctime)
    γparametric(t, p) = p[1] + p[2] * (t - t0) + p[3] * (t - t0)^2
   
    function Fbau!(du, u, p, t)
        du[1] = γparametric(t, p)
    end

    gcalib_problem = ODEProblem(Fbau!, [gcalib_data[1]], extrema(ipcctime))
    cost = build_loss_objective(
        gcalib_problem, Tsit5(), L2Loss(ipcctime, gcalib_data), 
        Optimization.AutoForwardDiff();
        maxiters = 10000, verbose = false
    )

    optprob = Optimization.OptimizationProblem(cost, zeros(3))
    calibratedγ = solve(optprob, BFGS())

    γᵇ(t) = γparametric(t, calibratedγ.u)

    # Initial mₛ
    baseidx = findfirst(==(0), ipcctime)
    N₀ = δₘ⁻¹(Eᵇ[baseidx] / Mᵇ[baseidx] - γᵇ(0), baseline)
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

begin
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

begin
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

begin
    
end