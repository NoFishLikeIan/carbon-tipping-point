using UnPack
using DifferentialEquations
using JLD2
using Interpolations, Dierckx

using StatsBase

using Plots, Printf

default(
    size = 800 .* (√2, 1), dpi = 320, 
    margins = 5Plots.mm, 
    linewidth = 1.5, thickness_scaling = 1.5
)

PLOTPATH = "plots"
SAVEFIG = true 
CONSTRAINED = true

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

# This assumes that all simulations have the same limits in (x, c)
@unpack Γ = first(simulationresults)

xₗ, xᵤ = extrema(Γ[1])
cₗ, cᵤ = extrema(Γ[2])
X = range(xₗ, xᵤ; length = 201)
C = range(cₗ, cᵤ; length = 201)

vuncon, euncon, vconst, econst = extractpoliciesfromsim(simulationresults)
e = CONSTRAINED ? econst : euncon # Use constrained emissions
v = CONSTRAINED ? vconst : vuncon # Use constrained emissions

σₓ = 0.5
γ = 26.75

# -- Climate dynamics plots
m = MendezFarazmand()
nullclinecarbon = (x -> nullcline(x, m)).(X)

begin # Albedo plot
    a(x) = g(x, m) + m.η * x^4

    albedofig = plot(
        X, a; 
        xlabel = "\$x\$, temperature deviations", 
        ylabel = "Energy input given albedo effect \$a(x)\$, (\$W/ m^{2}\$)", 
        c = :black, xticks = makedevxlabels(0., 15, m; step = 1, digits = 0), xlims = (xpreindustrial, xpreindustrial + 15), legend = false
    )

    vline!(albedofig, [m.x₀]; linewidth = 1.5, linestyle = :dash, c = :black)

    scatter!(albedofig, [m.x₀], [a(m.x₀)]; c = :black)

    if SAVEFIG  savefig(joinpath(PLOTPATH, "albedo.png")) end
    albedofig
end

begin # Business as Usual, ensemble simulation
    T = 80
    e₀ = 3. + m.δ * m.c₀
    
    function Fₑ!(du, u, e, t)
        x, c = u
        du[1] = μ(x, c, m)
        du[2] = e - m.δ * c
    end

    function Gₑ!(du, u, p, t)
        du[1] = σₓ
        du[2] = 0.
    end

    problembse = SDEProblem(Fₑ!, Gₑ!, [m.x₀, m.c₀], (0, T), e₀)
    ensemblebse = EnsembleProblem(problembse)

    bausim = solve(ensemblebse, SRIW1(), trajectories = 1000)
end

begin # BaU figure
    baulowerq, baumedian, bauupperq = extractquartiles(bausim, 0.1)
    xupper = 298

    Tsim = length(baulowerq)
    yticks = makedevxlabels(0., xupper - xpreindustrial, m; step = 1, withcurrent = true)

    bsefig = plot(
        nullclinecarbon, X;
        c = :black, linestyle = :dash, 
        ylabel = "\$x\$, temperature deviations",
        xlabel = "\$c\$, carbon concentration",
        xlims = (400, 600), ylims = (xₗ, xupper),
        yticks = yticks, linewidth = 2, label = false
    )

    
	plot!(bsefig, baumedian[:, 2], baumedian[:, 1];  fillrange = bauupperq, c = :darkred, fillalpha = 0.3, label = "BaU")
    plot!(bsefig, baumedian[:, 2], baumedian[:, 1]; linewidth = 0, fillrange = baulowerq, c = :darkred, fillalpha = 0.3, label = false)

    scatter!(bsefig, baumedian[1:(Tsim ÷ 10):end, 2], baumedian[1:(Tsim ÷ 10):end, 1]; c = :darkred, label = false)

	scatter!(bsefig, [m.c₀], [m.x₀]; c = :black, label = false)

    if SAVEFIG  savefig(joinpath(PLOTPATH, "sim-bse.png")) end

    bsefig
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

    function plotemissionssurface(σₓ; xᵤ = 6, cᵤ = 1000)
        temperatureticks = makedevxlabels(0, xᵤ, m; step = 1, digits = 0)

        efig = wireframe(
            range(xₗ, xᵤ + xpreindustrial; length = 61),
            range(cₗ, cᵤ; length = 61), 
            (x, c) -> e(x, c, σₓ, γ); 
            xlabel = "Temperature", zlabel = "Optimal emissions", ylabel = "CO\$_2\$ (p.p.m.)",
            xlims = (xₗ, xᵤ + xpreindustrial), ylims = (cₗ, cᵤ), 
            xticks = temperatureticks, legend = true,
            title = "Temperature variance, \$\\sigma^2_x = $(round(σₓ, digits = 2))\$",
            zlims = (-100, 100), camera = (45, 21),
            xlabelfontsize = 9, ylabelfontsize = 9, zlabelfontsize = 9,
            size = 600 .* (√2, 1)
        )

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

begin
    ensemblesim = simulateclimatepath(0.55, 26.85, m, e; T = T, ntraj = 1000) 

    ensembleemissions = computeoptimalemissions(0.7, 26.85, ensemblesim, e)

    plot(ensembleemissions, alpha = 0.05, c = :darkred, label = nothing)
end