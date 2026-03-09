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

includet("../utils.jl")
includet("../../../src/regret.jl")
includet("../../utils/approximate.jl")


## Define plotting constants
datapath = "data"

SAVEFIG = true;
PLOTPATH = "../job-market-paper/jeem/plots"
plotpath = joinpath(PLOTPATH, "regret")
if !isdir(plotpath) mkpath(plotpath) end

regretpolicypath = joinpath(datapath, "regret", "policy.jld2")
JLD2.@load regretpolicypath weights policybasis
weights = SVector{size(policybasis)}(weights)

