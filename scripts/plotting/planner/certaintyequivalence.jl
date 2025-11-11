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
plotpath = joinpath(PLOTPATH, abatementtype)
if !isdir(plotpath) mkpath(plotpath) end

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
    cefiles = listfiles(CEPATH)
    simulationfiles = listfiles(DATAPATH)

    thresholds = 2:0.05:4;
    discoveries = -1:0.25:1

    models = IAM[]
    valuefunctions = Dict{IAM, NTuple{2, ValueFunction}}()

end;