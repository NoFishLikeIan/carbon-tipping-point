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
    SAVEFIG = false 
end

include("../src/model/climate.jl")
include("../src/model/economic.jl")

include("../src/utils/dynamicalsystems.jl")

include("../src/utils/plotting.jl")
include("../src/utils/extractsim.jl")

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
    baseline = Hogg(σ²ₓ = 0.1)
    climate = (baseline, albedo)

    xₗ, xᵤ = baseline.xₚ, baseline.xₚ + 13
    
    mₗ, mᵤ = baseline.mₚ, mstable(xₗ, climate)
    
    X = range(xₗ, xᵤ; length = 201)
    M = range(mₗ, mᵤ; length = 201)
    
    nullclinecarbon = (x -> mstable(x, climate)).(X)
    # v, e = extractpoliciesfromsim(simulationresults)

    economy = Ramsey()
end

# -- Climate dynamics plots

begin # Albedo plot

    Δxᵤ = last(X) - baseline.xₚ
    
    Δamap = [0.02, 0.06, 0.08] 
    seqpaletteΔa = generateseqpalette(length(Δamap))
    
    Xₚ = collect(X .- baseline.xₚ)
    temperatureticks = makedevxlabels(0., Δxᵤ, climate; step = 1, digits = 0)
    albedovariation = [(x -> a(x, Albedo(a₂ = albedo.a₁ - Δa))).(X) for Δa ∈ Δamap]


    albedofig = @pgf Axis(
        {
            width = raw"1\textwidth",
            height = raw"0.6\textwidth",
            grid = "both",
            xlabel = raw"Temperature deviations $x - x^{\mathtt{p}}$",
            ylabel = raw"Albedo coefficient $a(x)$",
            xticklabels = temperatureticks[2],
            xtick = 0:1:Δxᵤ,
            no_markers,
            ultra_thick,
            xmin = 0, xmax = Δxᵤ
        }
    )

    @pgf for (i, albedodata) in enumerate(albedovariation)
        curve = Plot(
            {color=seqpaletteΔa[i], ultra_thick}, 
            Coordinates(
                collect(zip(Xₚ, albedodata))
            )
        ) 

        legend = LegendEntry("$(Δamap[i])")

        push!(albedofig, curve, legend)
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "albedo.tikz"), albedofig; include_preamble = true) 
    end

    albedofig
end

begin # Nullcline plot
    nullclinevariation = [(x -> mstable(x, (baseline, Albedo(a₂ = albedo.a₁ - Δa)))).(X) for Δa ∈ Δamap]


    nullclinefig = @pgf Axis(
        {
            width = raw"0.7\textwidth",
            height = raw"0.7\textwidth",
            grid = "both",
            ylabel = raw"Temperature deviations $x - x^{\mathtt{p}}$",
            xlabel = raw"Carbon concentration $m$",
            xmax = 1200,
            xtick = 0:300:1200,
            yticklabels = temperatureticks[2],
            ytick = 0:1:Δxᵤ,
            ymin = 0, ymax = Δxᵤ,
            ultra_thick, 
        }
    )

    @pgf for (i, nullclinedata) in enumerate(nullclinevariation)
        coords = Coordinates(collect(zip(nullclinedata, Xₚ)))

        curve = Plot({color = seqpaletteΔa[i]}, coords) 

        legend = LegendEntry("$(Δamap[i])")

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

    mbau = bauscenario[:, "CO2 concentration"]
    xbau = bauscenario[:, "Temperature"]
    Ebau = (Gtonoverppm / 1e9) * bauscenario[:, "CO2 emissions"]

    # Calibrate g
    gcalib_data = Array(log.(mbau)')
    t0 = first(ipcctime)
    gp(t, p) = p[1] + p[2] * (t - t0) + p[3] * (t - t0)^2
   
    function Fbau!(du, u, p, t)
        du[1] = gp(t, p)
    end

    gcalib_problem = ODEProblem(Fbau!, [gcalib_data[1]], extrema(ipcctime))
    cost = build_loss_objective(
        gcalib_problem, Tsit5(), L2Loss(ipcctime, gcalib_data), 
        Optimization.AutoForwardDiff();
        maxiters = 10000, verbose = false
    )

    optprob = Optimization.OptimizationProblem(cost, zeros(3))
    gcalibrated = solve(optprob, BFGS())

    g(t) = gp(t, gcalibrated.u)

    # Initial mₛ
    baseidx = findfirst(==(0), ipcctime)
    mₛ₀ = δₘ⁻¹(Ebau[baseidx] / mbau[baseidx] - g(0), baseline)
end


function simulatebau(Δa; trajectories = 1000) # Business as Usual, ensemble simulation    
    αbau = (x, m̂) -> 0.
    baualbedo = Albedo(a₂ = albedo.a₁ - Δa)
    
    bauparameters = ((Hogg(), baualbedo), g, αbau)
    
    SDEFunction(F!, G!, mass_matrix = mass_matrix(baseline))
    
    problembse = SDEProblem(SDEFunction(F!, G!, mass_matrix = mass_matrix(baseline)), G!, [baseline.x₀, log(baseline.m₀), mₛ₀], (0, T), bauparameters)
    
    ensemblebse = EnsembleProblem(problembse)
    
    bausim = solve(ensemblebse, trajectories = trajectories)
    baunullcline = (x -> mstable(x, (baseline, baualbedo))).(X)
    
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
            ylabel = raw"Growth rate $g^{\mathtt{b}}_t$",
            xlabel = raw"Year",
            xtick = 0:20:T,
            xmin = 0, xmax = T,
            xticklabels = BASELINE_YEAR .+ (0:20:T),
            ultra_thick, xticklabel_style = {rotate = 45}
        }
    )   
    
    gdata = g.(yearlytime)
    coords = Coordinates(zip(yearlytime, gdata))

    markers = @pgf Plot({ only_marks, mark_options = {fill = gcolor, scale = 1.5, draw_opacity = 0}, mark_repeat = 10}, coords) 

    curve = @pgf Plot({color = gcolor}, coords) 

    push!(gfig, curve, markers)

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "gfig.tikz"), gfig; include_preamble = true) 
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
            ytick = 0:1:Δxᵤ,
            ymin = 0, ymax = Δxᵤ,
            xmin = baseline.mₚ, xmax = 1200,
            xtick = 200:100:1100,
            grid = "both"
        }
    )

    @pgf for (i, Δa) ∈ enumerate([0, 0.08])
        isfirst = i == 1
        Δaplots = []
        timeseriescolor = isfirst ? seqpaletteΔa[1] : seqpaletteΔa[end]
    
        # IPCC benchmark line
        ipccbau = @pgf Plot(
            {
                very_thick, 
                color = "black", 
                mark = "*", 
                mark_options = {scale = 1.5, draw_opacity = 0}, 
                mark_repeat = 2
            }, 
            Coordinates(zip(mbau[3:end], xbau[3:end]))
        )

        push!(Δaplots, ipccbau)
        
        push!(Δaplots, LegendEntry("SSP5 - Baseline"))


        # Data simulation
        bausim, baunullcline = simulatebau(Δa; trajectories = 20)
        baumedian = [timepoint_median(bausim, t) for t in yearlytime]
        baumedianm = @. exp([u[2] for u in baumedian])
        baumedianx = @. first(baumedian) - xpreindustrial


        # Nullcline
        push!(Δaplots,
            @pgf Plot({dashed, color = "black", ultra_thick, forget_plot},
                Coordinates(collect(zip(baunullcline, Xₚ)))
            )
        )

        mediancoords = Coordinates(zip(baumedianm, baumedianx))

        label = isfirst ? raw"$\Delta a = 0.02$" : raw"$\Delta a = 0.08$"

        @pgf begin
            push!(
                Δaplots,
                Plot({ultra_thick, color = timeseriescolor, opacity = 0.8},mediancoords),
                LegendEntry(label),
                Plot({only_marks, mark_options = {fill = timeseriescolor, scale = 1.5, draw_opacity = 0, fill_opacity = 0.8}, mark_repeat = 20, forget_plot, mark = "*"}, mediancoords)
            )
        end

        @pgf for sim in bausim
            path = sim.(yearlytime)

            mpath = @. exp([u[2] for u in path])
            xpath = @. first(path) - xpreindustrial

            push!(
                Δaplots, 
                Plot({forget_plot, color = timeseriescolor, opacity = 0.2},
                    Coordinates(zip(mpath, xpath)),
                )
            )
        end
        
        nextgroup = isfirst ? {xlabel = raw"Carbon concentration $m_t$", ylabel = raw"Temperature deviations $x_t - x^{\mathtt{p}}$"} : {xlabel = raw"Carbon concentration $m_t$"}

        push!(baufig, nextgroup, Δaplots...)
    end

    @pgf baufig["legend style"] = raw"at = {(0.4, 0.95)}"

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "baufig.tikz"), baufig; include_preamble = true) 
    end

    baufig
end

yearsofdensity = 10:10:80
densedomain = collect(0:0.1:12)

baupossim, _ = simulatebau(0.08; trajectories = 51)
decadetemperatures = [first(componentwise_vectors_timepoint(baupossim, t)) .- xpreindustrial for t in yearsofdensity]
dists = (x -> kde(x)).(decadetemperatures)
densities = [x -> pdf(d, x) for d in dists]

begin
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
            xlabel = raw"Temperature deviations $x_t - x^{\mathtt{p}}$",
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
    Lfig = Plots.plot(xlabel = "Temperature,  \$x [K]\$", legendtitle = "Transition rate \$x_a\$", ylabel = "\$L(x)\$")
    for xₐ ∈ [1, 1e-1]
        plot!(Lfig, X, x -> L(x, Albedo(xₐ = xₐ)); label = "\$$(xₐ)\$", linewidth = 2)
    end

    Lfig
end