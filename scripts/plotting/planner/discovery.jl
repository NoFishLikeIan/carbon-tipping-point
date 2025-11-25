using Revise
using JLD2, UnPack, DataStructures
using FastClosures
using Base.Threads
using SciMLBase
using Statistics
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations, ForwardDiff, Dierckx
using StaticArrays

using Model, Grid
using Random; Random.seed!(11148705);

using Plots, PGFPlotsX, Contour
using LaTeXStrings, Printf
using Colors, ColorSchemes

# pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE,
    raw"\usepgfplotslibrary{fillbetween}",
    raw"\usetikzlibrary{patterns}",
    raw"\usepackage{siunitx}",
    raw"\DeclareSIUnit{\ppm}{p.p.m.}",
    raw"\DeclareSIUnit{\CO}{\,CO_2e}",
    raw"\DeclareSIUnit{\output}{trillion US\mathdollar / year}",
    raw"\DeclareSIUnit{\shortoutput}{tr US\mathdollar / y}",
)

includet("../../../src/valuefunction.jl")
includet("../../../src/extend/model.jl")
includet("../../../src/extend/grid.jl")
includet("../../../src/extend/valuefunction.jl")
includet("../../../scripts/markov/certaintyequivalence.jl")
includet("../utils.jl")
includet("../../utils/saving.jl")
includet("../../utils/simulating.jl")

damagetype = BurkeHsiangMiguel;
withnegative = true
abatementtype = withnegative ? "negative" : "constrained"
DATAPATH = "data/simulation-dense"; @assert isdir(DATAPATH)
CEPATH = "data/ce/simulation-dense"; @assert isdir(CEPATH)

SAVEFIG = true;
PLOTPATH = "../job-market-paper/submission/plots"
plotpath = joinpath(PLOTPATH, abatementtype)
if !isdir(plotpath) mkpath(plotpath) end

horizon = 100.
tspan = (0., horizon)

begin # Load climate claibration
    climatepath = joinpath("data", "calibration", "climate.jld2")
    @assert isfile(climatepath) "Climate calibration file not found at $climatepath"
    climatefile = jldopen(climatepath, "r+")
    @unpack calibration = climatefile
    close(climatefile)
end

begin # Load linear model
    linearsolpath = joinpath(DATAPATH, "linear", "growth", "logseparable", "negative", "Linear_burke_RRA10,00.jld2")
    @assert ispath(linearsolpath) "The linear simulation path does not exist: $linearsolpath"

    linearsimulation = loadtotal(linearsolpath);
end

begin # Load true threshold model
    threshold = 2.
    thresholdkey = replace("T$(Printf.format(Printf.Format("%.2f"), threshold))", "." => ",")
    thresholdsolfile = "$(thresholdkey)_burke_RRA10,00.jld2"
    thresholdsolpath = joinpath(DATAPATH, "tipping", "growth", "logseparable", "negative", thresholdsolfile)

    @assert ispath(thresholdsolpath) "The specified simulation file does not exist: $thresholdsolpath"

    thresholdsimulation = loadtotal(thresholdsolpath);
end

begin
    discoveries = [-0.5, 0., 0.5]
    αs = Interpolations.Extrapolation[]
    discoverymodels = IAM[]
    valuebydiscovery = []

    for discovery in discoveries
        println("Building discovery $discovery.")
        values, model, G = discoveryvalues(discovery, thresholdsimulation, linearsimulation)
        _, αitp = buildinterpolations(values, G);

        push!(αs, αitp)
        push!(discoverymodels, model)
        push!(valuebydiscovery, values)
    end
end

begin # Load calibration
    calibrationpath = "data/calibration"

    # Load economic calibration
    abatementpath = joinpath(calibrationpath, "abatement.jld2")
    @assert isfile(abatementpath) "Abatement calibration file not found at $abatementpath"
    abatementfile = jldopen(abatementpath, "r+")
    @unpack abatement = abatementfile
    close(abatementfile)

    investments = Investment()
    damages = Kalkuhl()
    economy = Economy(investments = investments, damages = damages, abatement = abatement)

    # Load climate calibration
    climatepath = joinpath(calibrationpath, "climate.jld2")
    @assert isfile(climatepath) "Climate calibration file not found at $climatepath"
    climatefile = jldopen(climatepath, "r+")
    @unpack hogg, feedbacklower, feedback, feedbackhigher, decay = climatefile
    close(climatefile)
end

begin # Plot aesthetics
    discoverylabels = ["\$\\Delta T^d = $(d)\$" for d in discoveries]
    PALETTE = colorschemes[:grays]
    colors = reverse(get(PALETTE, range(0, 0.6; length=length(discoveries))))

    TEMPLABEL = raw"Temperature deviations $T_t \; [\si{\degree\Celsius}]$"
    LINE_WIDTH = 2.5

    _, model, G = thresholdsimulation
    Tspace, mspace = G.ranges

    yearlytime = range(tspan[1], tspan[2]; step=1.)
    T₀ = hogg.T₀
    m₀ = log(hogg.M₀ / hogg.Mᵖ)
    x₀ = Point(T₀, m₀)

    temperatureticks = makedeviationtickz(Tspace[1], Tspace[end]; step=1, digits=2)

    X₀ = SVector(T₀, m₀, 0.)
    u₀ = SVector(X₀..., 0., 0., 0.) # Introduce three 0s for costs
end;

# -- Make simulation of optimal trajectories with discovery policies
begin
    simulations = Dict{Float64,EnsembleSolution}()
    _, model, _ = thresholdsimulation

    for (i, (discovery, αitp)) in enumerate(zip(discoveries, αs))
        parameters = (model, calibration, αitp);

        problem = SDEProblem(F, noise, u₀, tspan, parameters)
        ensembleprob = EnsembleProblem(problem)

        simulation = solve(ensembleprob; trajectories = 10_000)
        println("Done with discovery $discovery simulation $i / $(length(discoveries))")

        simulations[discovery] = simulation
    end
end

begin
    simfig = @pgf GroupPlot({
        group_style = {
            group_size = "$(length(discoveries)) by 2",
            horizontal_sep = raw"1em"
    }})

    yearticks = 0:20:horizon

    medianopts = @pgf {line_width = LINE_WIDTH}
    confidenceopts = @pgf {draw = "none", forget_plot}
    fillopts = @pgf {fill = "gray", opacity = 0.5}
    figopts = @pgf {width = raw"0.33\textwidth", height = raw"0.3\textwidth", grid = "both", xmin = 0, xmax = horizon}

    qs = (0.1, 0.5, 0.9)
    temperatureticks = makedeviationtickz(1, 3; step=0.5, digits=1)
    
    _, linearmodel, _ = linearsimulation
    mnpprob = ODEProblem((_, calibration, t) -> γ(t, calibration), m₀, (0, horizon), calibration)
    mnp = solve(mnpprob, Tsit5())
    mnppath = mnp(0:horizon)
    Mnppath = @. exp(mnppath) * hogg.Mᵖ

    # Carbon concentration in first row
    for (k, discovery) in enumerate(discoveries) 
        ensemble = simulations[discovery]
        paths = EnsembleAnalysis.timeseries_point_quantile(ensemble, qs, 0:horizon)
        mpaths = getindex.(paths.u, 2)
        Mpaths = [@. hogg.Mᵖ * exp(m) for m in mpaths]

        Mmedianplot = @pgf Plot(medianopts, Coordinates(0:horizon, getindex.(Mpaths, 2)))
        Mlowerplot = @pgf Plot({confidenceopts..., name_path = "Mlower"}, Coordinates(0:horizon, getindex.(Mpaths, 1)))
        Mupperplot = @pgf Plot({confidenceopts..., name_path = "Mupper"}, Coordinates(0:horizon, getindex.(Mpaths, 3)))

        Mfill = @pgf Plot(fillopts, raw"fill between [of=Mlower and Mupper]")

        labeloption = @pgf k > 1 ? { yticklabel = raw"\empty" } : { ylabel = L"\footnotesize Concentration $M_t \; [\si{ppm}]$" }

        nppath = @pgf Plot({ dashed, color = "gray", line_width = LINE_WIDTH }, Coordinates(0:horizon, Mnppath.u))

        @pgf push!(simfig, {figopts...,
            xticklabel = raw"\empty", ymin = hogg.M₀, ymax = 600.,
            title = discoverylabels[k], labeloption...,
        }, Mmedianplot, Mlowerplot, Mupperplot, Mfill, nppath)
    end

    # Temperature in second row
    for (k, discovery) in enumerate(discoveries)
        ensemble = simulations[discovery]

        paths = EnsembleAnalysis.timeseries_point_quantile(ensemble, qs, 0:horizon)
        Tpaths = first.(paths.u)

        Tmedianplot = @pgf Plot(medianopts, Coordinates(0:horizon, getindex.(Tpaths, 2)))
        Tlowerplot = @pgf Plot({confidenceopts..., name_path = "Tlower"}, Coordinates(0:horizon, getindex.(Tpaths, 1)))
        Tupperplot = @pgf Plot({confidenceopts..., name_path = "Tupper"}, Coordinates(0:horizon, getindex.(Tpaths, 3)))

        Tfill = @pgf Plot(fillopts, raw"fill between [of=Tlower and Tupper]")

        _, model, _ = thresholdsimulation
        Tnppath = [ Tstable(m, model.climate) |> first for m in mnppath.u ]
        nppath = @pgf Plot({ dashed, color = "gray", line_width = LINE_WIDTH }, Coordinates(0:horizon, Tnppath))

        figticks = yearticks[1:(k < 3 ? end - 1 : end)]

        Toptionfirst = @pgf k > 1 ? {
            yticklabel = raw"\empty"
        } : {
            ytick = temperatureticks[1], yticklabels = temperatureticks[2],
            ylabel = L"\footnotesize Temperature $T_t \; [\si{\degree}]$"
        }

        @pgf push!(simfig, {figopts...,
                ymin = minimum(temperatureticks[1]),
                ymax = maximum(temperatureticks[1]),
                xtick = figticks,
                xticklabels = 2020 .+ Int.(figticks),
                xticklabel_style = {rotate = 45},
                xlabel = "Year",
                Toptionfirst...
            }, Tmedianplot, Tlowerplot, Tupperplot, Tfill, nppath)
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "discovery-simfig.tikz"), simfig; include_preamble=true)
    end

    simfig
end