using Revise
using JLD2, UnPack, DataStructures
using FastClosures
using Base.Threads
using SciMLBase
using Statistics
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations, Dierckx
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

begin # Load climate claibration
    climatepath = joinpath("data", "calibration", "climate.jld2")
    @assert isfile(climatepath) "Climate calibration file not found at $climatepath"
    climatefile = jldopen(climatepath, "r+")
    @unpack calibration, hogg = climatefile
    close(climatefile)
end

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
    x₀ = Point(hogg.T₀, log(hogg.M₀ / hogg.Mᵖ))
    horizon = 80.
    simulationfiles = listfiles(DATAPATH)

    thresholds = 2:0.05:4
    discoveries = -1:0.25:1

    truevalue = fill(NaN, length(thresholds))
    discoveryvalue = fill(NaN, length(thresholds), length(discoveries))

    for (i, threshold) in enumerate(thresholds)
        @printf("Loading threshold=%.2f\r", threshold)

        filepath = findthreshold(threshold, simulationfiles)
        optimalvalue, _, G = loadtotal(filepath; tspan = (0., 1.2horizon))
        Hopt, _ = buildinterpolations(optimalvalue, G);
        H₀ = Hopt(x₀.T, x₀.m, 0.)

        truevalue[i] = H₀

        for (j, discovery) in enumerate(discoveries)

            thresholdkey = replace("T$(Printf.format(Printf.Format("%.2f"), threshold))", "." => ",")
            discoverykey = replace("D$(Printf.format(Printf.Format("%.2f"), discovery))", "." => ",")
            outfile = joinpath(CEPATH, "$(thresholdkey)_$(discoverykey).jld2")

            if !isfile(outfile)
                @warn "Outfile $outfile not found!"
                continue
            end

            JLD2.@load outfile valuefunction
            Hᵈ₀ = interpolateovergrid(valuefunction.H, G, x₀)
            discoveryvalue[i, j] = Hᵈ₀
        end
    end
end;

begin # CE surface
    
end