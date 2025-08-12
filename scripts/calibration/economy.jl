using Revise

using DotEnv, UnPack
using CSV, DataFrames, JLD2

using DifferentialEquations
using DiffEqParamEstim, Optimization, OptimizationOptimJL

using Model

env = DotEnv.config()

economy = Economy(τ = calibration.τ)

