using UnPack

using Dierckx
using LinearAlgebra

using DifferentialEquations
using DifferentialEquations.EnsembleAnalysis
using DiffEqParamEstim, Optimization, OptimizationOptimJL

using CSV, DataFrames, JLD2

using Plots, Printf, PGFPlotsX, Colors

begin # Global variables
    PALETTE = color.(["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"])

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
    
    a₂map = [Albedo().a₂, 0.23, 0.28]
    
    Xₚ = collect(X .- baseline.xₚ)
    temperatureticks = makedevxlabels(0., Δxᵤ, climate; step = 1, digits = 0)
    albedovariation = [(x -> a(x, Albedo(a₂ = a₂))).(X) for a₂ ∈ a₂map]


    axis = @pgf Axis(
        {
            width = raw"1\textwidth",
            height = raw"0.6\textwidth",
            grid = "both",
            xlabel = "Temperature deviations",
            ylabel = raw"Albedo coefficient $a(x)$",
            xticklabels = temperatureticks[2],
            xtick = 0:1:Δxᵤ,
            no_markers,
            very_thick,
            xmin = 0, xmax = Δxᵤ
        }
    )

    @pgf for (i, albedodata) in enumerate(albedovariation)
        curve = Plot(
            {color=PALETTE[i]}, 
            Coordinates(
                collect(zip(Xₚ, albedodata))
            )
        ) 

        legend = LegendEntry("$(a₂map[i])")

        push!(axis, curve, legend)
    end

    if SAVEFIG PGFPlotsX.save(joinpath(PLOTPATH, "albedo.tikz"), axis; include_preamble = true) end

    axis
end

begin # Nullcline plot
    nullclinevariation = [(x -> mstable(x, (baseline, Albedo(a₂ = a₂)))).(X) for a₂ ∈ a₂map]


    nullclinefig = @pgf Axis(
        {
            width = raw"0.7\textwidth",
            height = raw"0.7\textwidth",
            grid = "both",
            ylabel = raw"Temperature deviations $x$",
            xlabel = raw"Carbon concentration $m$",
            xmax = 1200,
            xtick = 0:300:1200,
            yticklabels = temperatureticks[2],
            ytick = 0:1:Δxᵤ,
            ymin = 0, ymax = Δxᵤ,
            very_thick, 
        }
    )

    @pgf for (i, nullclinedata) in enumerate(nullclinevariation)
        coords = Coordinates(collect(zip(nullclinedata, Xₚ)))

        curve = Plot({color = PALETTE[i]}, coords) 

        legend = LegendEntry("$(a₂map[i])")

        push!(nullclinefig, curve, legend)
    end

    initscatter = @pgf Plot({mark = "*", color = "black"},
        Table(
            x = [baseline.m₀],
            y = [baseline.x₀ - xpreindustrial],
        )
    )

    push!(nullclinefig, initscatter)

    if SAVEFIG PGFPlotsX.save(joinpath(PLOTPATH, "nullcline.tikz"), nullclinefig; include_preamble = true) end

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

function simulatebau(a₂; trajectories = 1000) # Business as Usual, ensemble simulation    
    αbau = (x, m̂) -> 0.
    baualbedo = Albedo(a₂ = a₂)

    bauparameters = ((Hogg(), baualbedo), g, αbau)

    SDEFunction(F!, G!, mass_matrix = mass_matrix(baseline))

    problembse = SDEProblem(SDEFunction(F!, G!, mass_matrix = mass_matrix(baseline)), G!, [baseline.x₀, log(baseline.m₀), mₛ₀], (0, T), bauparameters)

    ensemblebse = EnsembleProblem(problembse)

    bausim = solve(ensemblebse, trajectories = trajectories)
    baunullcline = (x -> mstable(x, (baseline, baualbedo))).(X)

    return bausim, baunullcline
end

bausim, baunullcline = simulatebau(0.21; trajectories = 30)

begin # BaU figure
    yearlytime = 0:1:T
    medianbau = [timepoint_median(bausim, t) for t in yearlytime]

    baumedianm = @. exp([u[2] for u in medianbau])
    baumedianx = @. first(medianbau) - xpreindustrial

    mediancolor = PALETTE[end - 1]

    baufig = @pgf Axis(
        {
            width = raw"1\textwidth",
            height = raw"0.707\textwidth",
            grid = "both",
            ylabel = raw"Temperature deviations $x$",
            xlabel = raw"Carbon concentration $m$",
            yticklabels = temperatureticks[2],
            ytick = 0:1:Δxᵤ,
            ymin = 0, ymax = Δxᵤ,
            xmin = baseline.mₚ, xmax = 1200
        }
    )

    

    ipccbau = @pgf Plot({ultra_thick, color = "black", mark = "*"}, Coordinates(zip(mbau[3:end], xbau[3:end])))

    baulegend = @pgf LegendEntry("SSP5 - Baseline")

    push!(baufig, ipccbau, baulegend)

    push!(baufig,
        @pgf Plot({dashed, color = "black", ultra_thick, forget_plot},
            Coordinates(collect(zip(baunullcline, Xₚ)))
        )
    )

    mediancoords = Coordinates(zip(baumedianm, baumedianx))
    medianbaufig = @pgf Plot({ultra_thick, color = mediancolor}, mediancoords)
    medianscatter = @pgf Plot({only_marks, mark_options = {fill = mediancolor}, mark_repeat = 10, forget_plot}, mediancoords)

    push!(baufig, medianbaufig, LegendEntry("Business-as-usual"), medianscatter)


    @pgf for (i, sim) in enumerate(bausim)
        path = sim.(yearlytime)

        coords = Coordinates(collect(zip(
            exp.([u[2] for u in path]),
            first.(path) .- xpreindustrial    
        )))

        curve = Plot({opacity = 0.3, color = mediancolor}, coords) 

        push!(baufig, curve)
    end

    @pgf baufig["legend style"] = raw"at = {(0.4, 0.95)}"


    baufig
end