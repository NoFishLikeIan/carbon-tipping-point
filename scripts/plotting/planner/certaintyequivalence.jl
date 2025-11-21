using Revise
using JLD2, UnPack, DataStructures
using FastClosures
using Base.Threads
using SciMLBase
using Statistics
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations, ForwardDiff
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
includet("../utils.jl")
includet("../../utils/saving.jl")
includet("../../utils/simulating.jl")

CEPATH = "data/ce/simulation-dense"; @assert isdir(CEPATH)
DATAPATH = "data/simulation-dense"; @assert isdir(DATAPATH)

SAVEFIG = true;
PLOTPATH = "../job-market-paper/submission/plots"
plotpath = joinpath(PLOTPATH, "negative")
if !isdir(plotpath) mkpath(plotpath) end

linearmodel, G = joinpath(DATAPATH, "linear/growth/logseparable/negative/Linear_burke_RRA10,00.jld2") |> loadproblem


begin # Plot estetics
    PALETTE = colorschemes[:grays]

    TEMPLABEL = raw"Temperature deviations $T_t \; [\si{\degree\Celsius}]$"
    LINE_WIDTH = 2.5

    Tspace, mspace = G.ranges

    T₀ = linearmodel.climate.hogg.T₀
    m₀ = log(linearmodel.climate.hogg.M₀ / linearmodel.climate.hogg.Mᵖ)
    x₀ = Point(T₀, m₀)

    temperatureticks = makedeviationtickz(Tspace[1], Tspace[end]; step=1, digits=2)

    X₀ = SVector(T₀, m₀, 0.)
    u₀ = SVector(X₀..., 0., 0., 0.) # Introduce three 0s for costs
end;

function findthreshold(threshold, simulationfiles)
    for filepath in simulationfiles
        model = loadproblem(filepath) |> first
        abatementdir = splitpath(filepath)[end - 1]

        istype = (model.economy.damages isa BurkeHsiangMiguel) && (abatementdir == "negative")
        isthreshold = model.climate isa TippingClimate && model.climate.feedback.Tᶜ ≈ threshold


        if istype && isthreshold
            return filepath
        end
    end

    return nothing
end

begin # Import available simulation and CE files
    horizon = 100.
    simulationfiles = listfiles(DATAPATH)

    thresholds = 2:0.05:4;
    discoveries = -1:0.05:1

    truevalue = fill(NaN, length(thresholds))
    truegradient = fill(Point(NaN, NaN), size(truevalue))

    discoveryvalue = fill(NaN, length(thresholds), length(discoveries))
    discoverygradient = fill(Point(NaN, NaN), size(discoveryvalue))

    models = IAM[]
    interpolations = Dict{IAM, NTuple{2, Interpolations.Extrapolation}}()

    missingpairs = Dict{String, Float64}[];

    for (i, threshold) in enumerate(thresholds)
        @printf("Loading threshold=%.2f\r", threshold)

        filepath = findthreshold(threshold, simulationfiles)
        optimalvalues, model, G = loadtotal(filepath; tspan = (0., 1.2horizon))
        Hopt, αopt = buildinterpolations(optimalvalues, G);
        H₀ = Hopt(x₀.T, x₀.m, 0.)
        truevalue[i] = Hopt(x₀.T, x₀.m, 0.)
        truegradient[i] = ForwardDiff.gradient(x ->  Hopt(x[1], x[2], 0.), x₀)

        push!(models, model)
        interpolations[model] = (Hopt, αopt);

        for (j, discovery) in enumerate(discoveries)

            thresholdkey = replace("T$(Printf.format(Printf.Format("%.2f"), threshold))", "." => ",")
            discoverykey = replace("D$(Printf.format(Printf.Format("%.2f"), discovery))", "." => ",")
            outfile = joinpath(CEPATH, "$(thresholdkey)_$(discoverykey).jld2")

            if !isfile(outfile)
                @warn "Outfile $outfile not found!"

                push!(missingpairs, Dict("threshold" => threshold, "discovery" => discovery))

                continue
            end

            JLD2.@load outfile H₀ ∇H₀
            discoveryvalue[i, j] = H₀
            discoverygradient[i, j] = ∇H₀
        end
    end

    linearfilepath = joinpath(DATAPATH, "linear/growth/logseparable/negative/Linear_burke_RRA10,00.jld2")
    linearvalues, linearmodel, linearG = loadtotal(linearfilepath; tspan = (0., 1.2horizon))

    push!(models, linearmodel)
    interpolations[linearmodel] = buildinterpolations(linearvalues, linearG)
end;


