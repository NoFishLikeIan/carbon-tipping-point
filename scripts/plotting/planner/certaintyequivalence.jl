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

damagetype = BurkeHsiangMiguel;
withnegative = true
abatementtype = withnegative ? "negative" : "constrained"
CEPATH = "data/ce/simulation-dense"; @assert isdir(CEPATH)
DATAPATH = "data/simulation-dense"; @assert isdir(DATAPATH)

SAVEFIG = true;
PLOTPATH = "../job-market-paper/submission/plots"
plotpath = joinpath(PLOTPATH, abatementtype)
if !isdir(plotpath) mkpath(plotpath) end

discoveries = -1:0.25:1

begin # Read available simulation files
    simulationfiles = listfiles(DATAPATH)
    nfiles = length(simulationfiles)
    G = simulationfiles |> first |> loadproblem |> last
    
    modelfiles = String[]
    for (i, filepath) in enumerate(simulationfiles)
        print("Reading $i / $(length(simulationfiles))\r")
        model, _ = loadproblem(filepath)
        abatementdir = splitpath(filepath)[end - 1]

        if (model.economy.damages isa damagetype) && (abatementdir == abatementtype)
            push!(modelfiles, filepath)
        end
    end
end;


begin # Import available simulation and CE files
    cefiles = listfiles(CEPATH)

    models = IAM[]
    valuefunctions = Dict{IAM, NTuple{2, ValueFunction}}()
    
    for (i, filepath) = enumerate(modelfiles)
        print("Loading $i / $(length(modelfiles))\r")
        values, model, _ = loadtotal(filepath; tspan=(0, 0.05))

        if model.climate isa LinearClimate continue end
            
        for 
        
        truevalue =  values[0.0]
        valuefunctions[model] = (truevalue, )
        push!(models, model)
    end

    sort!(by = m -> m.climate, models)

    cefiles = listfiles(CEPATH)
    nfiles = length(cefiles)

    discoveryvaluefunctions = Tuple{Float64, Float64, ValueFunction}[]

    for cefile in cefiles
        JLD2.@load cefile threshold discovery valuefunction
        push!(discoveryvaluefunctions, (threshold, discovery, valuefunction))
    end
end;