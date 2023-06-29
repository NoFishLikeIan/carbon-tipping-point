using UnPack
using DifferentialEquations
using JLD2
using Interpolations, Dierckx

using KernelDensity

using CSV, DataFrames

using StatsBase

using Plots, Printf

default(
    size = 800 .* (√2, 1), dpi = 320, 
    margins = 5Plots.mm, 
    linewidth = 1.5, thickness_scaling = 1.5
)

BASELINE_YEAR = 2020

PLOTPATH = "plots"
DATAPATH = "data/climate-data"
SAVEFIG = false 
CONSTRAINED = false

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

begin # Load the latest simulation
    datapath = "data/sims"; @assert ispath(datapath)
    availablesims = map(gettimestamp, readdir(datapath))

    data = load(joinpath(datapath, "valuefunction_$(maximum(availablesims)).jld2"))
    parameters = get(data, "parameters",  [])
    simulationresults = get(data, "solution",  [])
end

begin # This assumes that all simulations have the same limits in (x, c)
    @unpack Γ = first(simulationresults)

    climate = MendezFarazmand()
    xₗ, xᵤ = climate.xₚ, climate.xₚ + 13
    cₗ, cᵤ = extrema(Γ[2])
    X = range(xₗ, xᵤ; length = 201)
    C = range(cₗ, cᵤ; length = 201)

    v, e = extractpoliciesfromsim(simulationresults)

    economy = Ramsey()
    σ²ₓ = 0.2
end

# -- Climate dynamics plots
nullclinecarbon = (x -> nullcline(x, climate)).(X)

begin # Albedo plot
    a(x) = g(x, climate) + climate.η * x^4

    albedofig = plot(
        X, a; 
        xlabel = "\$x\$, temperature deviations", 
        ylabel = "Energy input given albedo effect \$a(x)\$, (\$W/ m^{2}\$)", 
        c = :black, xticks = makedevxlabels(0., 15, climate; step = 1, digits = 0), xlims = (xpreindustrial, xpreindustrial + 15), legend = false
    )

    vline!(albedofig, [climate.x₀]; linewidth = 1.5, linestyle = :dash, c = :black)

    scatter!(albedofig, [climate.x₀], [a(climate.x₀)]; c = :black)

    if SAVEFIG  savefig(joinpath(PLOTPATH, "albedo.png")) end
    albedofig
end

ipccdatapath = joinpath(DATAPATH, "proj-median.csv")
ipccproj = CSV.read(ipccdatapath, DataFrame)
ssp3data = filter(:Scenario => isequal("SSP3 - Baseline"), ipccproj)

ipccemissions = @. ssp3data[:, "CO2 emissions"] * Gtonoverppm / 1e9
timeproj = ssp3data.Year
T = 80
timesimulation = range(0, T; length = 101) .+ BASELINE_YEAR
ssp3median = ssp3data.Temperature .+ xpreindustrial

ssp1median = filter(:Scenario => isequal("SSP1 - 1.9"), ipccproj).Temperature .+ xpreindustrial

itp = linear_interpolation(timeproj .- timeproj[1], ipccemissions, extrapolation_bc = Line())
eᵢ = t -> itp(t)

begin # Business as Usual, ensemble simulation
    e₀ = economy.e₀
    
    function Fₑ!(du, u, p, t)
        x, m = u
        du[1] = μ(x, m, climate)
        du[2] = eᵢ(t) - climate.δ * m
    end

    function Gₑ!(du, u, p, t)
        du[1] = σ²ₓ
        du[2] = 0.
    end

    problembse = SDEProblem(Fₑ!, Gₑ!, [climate.x₀, climate.m₀], (0, T), e₀)
    ensemblebse = EnsembleProblem(problembse)

    bausim = solve(ensemblebse, SRIW1(), trajectories = 1000)
end

begin # BaU figure
    baulowerq, baumedian, bauupperq = extractquartiles(bausim, 0.1)
    xupper = 298

    Tsim = length(baulowerq)
    bauticks = makedevxlabels(0., xupper - xpreindustrial, climate; step = 1, withcurrent = true)

    bsefig = plot(
        nullclinecarbon, X;
        c = :black, linestyle = :dash, 
        ylabel = "\$x\$ temperature",
        xlabel = "\$m\$ carbon concentration",
        xlims = (minimum(nullclinecarbon), maximum(baumedian[:, 2])), ylims = (climate.xₚ, xupper),
        yticks = bauticks, 
        linewidth = 2, label = false
    )
    
    for simulation in bausim
        pathi = simulation(range(0, T; length = 101))
        
        plot!(bsefig, last.(pathi.u), first.(pathi.u); label = false, c = :darkred, alpha = 0.01)
    end
    plot!(bsefig, baumedian[:, 2], baumedian[:, 1]; c = :darkred, label = "Model with SSP3 - Baseline emissions", linewidth = 3)

    scatter!(bsefig, baumedian[1:(Tsim ÷ 10):end, 2], baumedian[1:(Tsim ÷ 10):end, 1]; c = :darkred, label = false)

	scatter!(bsefig, [climate.m₀], [climate.x₀]; c = :darkred, label = false)

    if SAVEFIG  savefig(bsefig, joinpath(PLOTPATH, "sim-bse.png")) end

    bsefig
end


begin # BaU time

    ipccfig = plot(
        timeproj, ssp3median;
        c = :black, marker = :o, 
        ylabel = "\$x\$ temperature deviations",
        xlabel = "Year",
        yticks = bauticks, linewidth = 2, 
        label = "SSP3 - Baseline temperature"
    )
    
    for simulation in bausim
        pathi = simulation(range(0, T; length = 101))
        
        plot!(ipccfig, timesimulation, first.(pathi.u); label = false, c = :darkred, alpha = 0.01)
    end

    scatter!(ipccfig, [BASELINE_YEAR], [climate.x₀]; c = :darkred, label = false)

    if SAVEFIG  savefig(ipccfig, joinpath(PLOTPATH, "ipcc-compare.png")) end

    ipccfig
end

begin # Damage function
    damagefig = plot(
        X, x -> d(x, economy); 
        xlabel = "\$x\$ temperature deviations", 
        ylabel = "\$d(x)\$ damage function", 
        c = :black, xticks = makedevxlabels(0., 15, climate; step = 1, digits = 0), xlims = (xpreindustrial, xpreindustrial + 12), legend = false
    )

    vline!(damagefig, [climate.x₀]; linewidth = 1.5, linestyle = :dash, c = :black)

    scatter!(damagefig, [climate.x₀], [d(climate.x₀, economy)]; c = :black)

    if SAVEFIG  savefig(damagefig, joinpath(PLOTPATH, "damagefig.png")) end
    damagefig
end

begin # Distribution of damages
    timedamagedens = 10:10:50
    D = Matrix{Float64}(undef, length(timedamagedens), length(bausim))

    for (i, simulation) in enumerate(bausim)
        pathi = simulation(timedamagedens)
        
        D[:, i] = (x -> d(x, economy)).(first.(pathi.u))
    end
end

begin
    damagedensities = kde.(eachrow(D))
    damagedensitiesfig = plot(xlabel = "Damage distribution", legendtitle = "Year")

    damagecolors = palette(:coolwarm, length(timedamagedens))

    unit = range(0, 1; length = 1001)

    for (i, t) in enumerate(timedamagedens)
        idens = (d -> pdf(damagedensities[i], d)).(unit)
        plot!(damagedensitiesfig, unit, idens ./ sum(idens), label = Int(BASELINE_YEAR + t), c = damagecolors[i], linewidth = 3)
    end

    if SAVEFIG  savefig(damagedensitiesfig, joinpath(PLOTPATH, "damagedensitiesfig.png")) end

    damagedensitiesfig
end

# -- Value function plots
# Optimal paths

begin # Emission deviations
    frames = 60
    gifσspace = [
        range(0, 1; length = frames)..., 
        ones(frames ÷ 4)..., 
        reverse(range(0, 1; length = frames))...,
        zeros(frames ÷ 4)...
    ]

    function plotemissionssurface(σₓ; xᵤ = 6, cᵤ = 1000, l = 20)
        temperatureticks = makedevxlabels(0, xᵤ, climate; step = 1, digits = 0)
        temp = range(xₗ, xᵤ + xpreindustrial; length = 31)
        carbon = range(cₗ, cᵤ; length = 31)

        efig = wireframe(
            temp, carbon, (x, c) -> e(x, c, σₓ); 
            xlabel = "Temperature", zlabel = "Optimal emissions", ylabel = "CO\$_2\$ (p.p.m.)",
            xlims = (xₗ, xᵤ + xpreindustrial), ylims = (cₗ, cᵤ), 
            xticks = temperatureticks, legend = true,
            title = "Temperature variance, \$\\sigma^2_x = $(round(σₓ, digits = 2))\$",
            camera = (45, 21),
            xlabelfontsize = 9, ylabelfontsize = 9, zlabelfontsize = 9,
            size = 600 .* (√2, 1), levels = l, c = :coolwarm
        )

        surface!(efig, temp, carbon, (x, c) -> e(x, c, σₓ), alpha = 0., c = :coolwarm, colorbar = false)

        efig
    end

    if SAVEFIG
        anim = @animate for (i, σₓ) ∈ enumerate(gifσspace)
            print("Building frame $i / $(length(gifσspace))\r")
            plotemissionssurface(σₓ)
        end

        gif(anim, joinpath(PLOTPATH, "emission-deviations.gif"), fps = 15)
    end
end

ensemblesim = simulateclimatepath(σ²ₓ, climate, e; T = T, ntraj = 1000) 
begin

    ensembleemissions = computeoptimalemissions(σ²ₓ, ensemblesim, e; Tsim = length(timesimulation))

    optemissionfig = plot(timesimulation, ensembleemissions, alpha = 0.01, c = :darkblue, label = nothing)
end

begin # Optimal temperature path

    opttemp = plot(
        timeproj, ssp3median;
        c = :black, marker = :o, 
        ylabel = "\$x\$ temperature deviations",
        xlabel = "Year",
        yticks = bauticks, linewidth = 2, 
        label = "SSP3 - Baseline temperature"
    )

    plot!(opttemp,
        timeproj, ssp1median;
        c = :blue, marker = :o, 
        label = "SSP1 - Baseline temperature"
    )
    
    for simulation in ensemblesim
        pathi = simulation(range(0, T; length = 101))
        
        plot!(opttemp, timesimulation, first.(pathi.u); label = false, c = :darkred, alpha = 0.01)
    end

    scatter!(opttemp, [BASELINE_YEAR], [climate.x₀]; c = :darkred, label = false)

    if SAVEFIG  savefig(opttemp, joinpath(PLOTPATH, "optimal-path.png")) end

    opttemp
end


begin # Consumption
    D = extractecondatafromsim(ensemblesim, e, climate, economy)
    consumptionplot = plot()

    for i in axes(D, 1)
        plot!(consumptionplot, first(ensemblesim).t, D[i, :, 2]; label = false, color = :darkred, alpha = 0.01)
    end

    consumptionplot
end

opttemp