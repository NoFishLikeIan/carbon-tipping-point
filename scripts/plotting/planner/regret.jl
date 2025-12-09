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

simpath = "data/simulations/simulation-dense"; @assert isdir(simpath)

SAVEFIG = true;
PLOTPATH = "../job-market-paper/jeem/plots"
plotpath = joinpath(PLOTPATH, "regret")
if !isdir(plotpath) mkpath(plotpath) end

function loadsimulation(threshold, discovery, simpath)
    thresholdkey = replace("T$(Printf.format(Printf.Format("%.2f"), threshold))", "." => ",")
    discoverykey = replace("D$(Printf.format(Printf.Format("%.2f"), discovery))", "." => ",")
    outfile = joinpath(simpath, "$(thresholdkey)_$(discoverykey).jld2")

    return JLD2.load_object(outfile)
end

begin # Labels, colors and axis
    PALETTE = colorschemes[:grays]
    colors = get(PALETTE, range(0., 1.; length = length(models)), (0., 1.25))

    colorsbymodel = Dict(models .=> colors)
    Tmin = 0.0; Tmax = 6.0
    Tspace = range(Tmin, Tmax; length = 101)

    horizon = 2100. - first(calibration.calibrationspan)
    yearlytime = 0:1:horizon
    simtspan = (0, horizon)

    temperatureticks = collect.(makedeviationtickz(0, 6; step=1, digits=0))

    m₀ = log(hogg.M₀ / hogg.Mᵖ)
    T₀ = hogg.T₀
    X₀ = SVector(T₀, m₀)
end;