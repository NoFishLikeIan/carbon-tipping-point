using Test, BenchmarkTools, Revise, UnPack
using Plots, LaTeXStrings
default(c = :viridis, label = false, dpi = 180)

using Model, Grid

using Base.Threads

using DataStructures
using SciMLBase
using ZigZagBoomerang
using Statistics
using StaticArrays
using FastClosures
using LinearAlgebra

using Optimization
using OptimizationOptimJL, LineSearches

using ForwardDiff

using JLD2
using Printf, Dates

includet("../../src/valuefunction.jl")
includet("../../src/extensions.jl")
includet("../utils/saving.jl")
includet("../utils/logging.jl")
includet("../markov/chain.jl")
includet("../markov/terminal.jl")
includet("../markov/backward.jl")

simfile = "data/simulation-large-lbfgs/constrained/tipping/growth/logseparable/Tc=289,15_ρ=0,01500_θ=10,00_ψ=1,00_σT=0,1442_σm=0,0078_ωr=0,01756_ξ1=0,035700_ξ2=0,001800.jld2"; @assert isfile(simfile)

states, G, model = loadtotal(simfile)