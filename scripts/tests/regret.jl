using BenchmarkTools, Revise, UnPack
using Plots, LaTeXStrings
default(c=:viridis, label=false, dpi=180)

using Model, Grid
using Base.Threads
using SciMLBase
using Statistics
using StaticArrays, SparseArrays
using Interpolations, FastChebInterp, DataStructures
using FastClosures

using Optimization, OptimizationOptimJL, OptimizationBase, Ipopt
using DifferentiationInterface, ADTypes, ForwardDiff

using LinearSolve, LinearAlgebra

using JLD2, UnPack
using Dates, Printf

includet("../../src/valuefunction.jl")
includet("../../src/extend/model.jl")
includet("../../src/extend/grid.jl")
includet("../../src/extend/valuefunction.jl")
includet("../../src/regret.jl")
includet("../utils/saving.jl")
includet("../utils/simulating.jl")
includet("../utils/loading.jl")
includet("../plotting/utils.jl")
includet("../utils/simulating.jl")
includet("../utils/approximate.jl")
includet("../markov/chain.jl")
includet("../markov/finitedifference.jl")
includet("../regret/chain.jl")
includet("../regret/finitedifference.jl")

## Construct the model
DATAPATH = "data"
calibrationpath = joinpath(DATAPATH, "calibration")

# Load economic calibration
abatementpath = joinpath(calibrationpath, "abatement.jld2")
@assert isfile(abatementpath) "Abatement calibration file not found at $abatementpath"
abatementfile = jldopen(abatementpath, "r+")
@unpack abatement = abatementfile
close(abatementfile)

investments = Investment()
damages = BurkeHsiangMiguel() # WeitzmanGrowth()
economy = Economy(investments = investments, damages = damages, abatement = abatement)

# Load climate claibration
climatepath = joinpath(calibrationpath, "climate.jld2")
@assert isfile(climatepath) "Climate calibration file not found at $climatepath"
climatefile = jldopen(climatepath, "r+")
@unpack calibration, hogg, feedbacklower, feedback, feedbackhigher = climatefile
close(climatefile)

## Initialise model
# Time
Δt⁻¹ = 12.
Δt = 1 / Δt⁻¹
τ = 500.

preferences = LogSeparable()
decay = ConstantDecay(0.)
climate = LinearClimate(hogg, decay)

preferences = LogSeparable()
linearmodel = IAM(climate, economy, preferences)

## Construct policy simplex
simpath = "data/simulation";
paths = loadsimulationpaths(simpath; exclude = ["terminal", "linear"])
K = 10;
_, G = loadproblem(paths[2.0])

filteredpaths = basispolicypaths(paths, K, G; tspan = (0, 10)) # Indices of the basis
policybasis = SimplexPolicies(filteredpaths);

## Construct Chebyshev reresentation of full information
values = OrderedDict(k => loadtotal(p) for (k, p) in paths)

order = (20, 20, 10, 5)
H̃ = chebyshevrepresentation(values, order);

## Compute regret
G = coarse(G, (4, 4))
valuefunction = ValueFunction(τ, climate, G, calibration)

weights = MVector{K}(rand(K)); 
weights ./= sum(weights)

function regret(weights, threshold, H̃, valuefunction, τ, Δt, (linearmodel, feedback), G, calibration, policybasis; steadystatetolerance = Error{eltype(G)}(1e-3, 1e-3), verbose = 0, problem = nothing)
    valuefunction.t.t = τ
    
    climate = TippingClimate(linearmodel.climate.hogg, linearmodel.climate.decay, updatethreshold(threshold, feedback))
    model = IAM(climate, linearmodel.economy, linearmodel.preferences)

    steadystate!(valuefunction, weights, Δt, model, G, calibration, policybasis; tolerance = steadystatetolerance, verbose = verbose)
    backwardsimulation!(valuefunction, weights, Δt, model, G, calibration, policybasis; verbose = verbose)

    x₀ = Point(climate.hogg.T₀, log(climate.hogg.M₀ / climate.hogg.Mᵖ))
    Gⱼ = interpolateovergrid(valuefunction.H, G, x₀)
    Hⱼ = H̃(SVector(x₀.T, x₀.m, zero(τ), threshold))

    return Gⱼ - Hⱼ
end

function maxregret(weights, optparameters)
    H̃, valuefunction, τ, Δt, (linearmodel, feedback), G, calibration, policybasis = optparameters

    # Initialize problem once (factor the operator)
    problem = initsteadystateproblem(valuefunction, weights, Δt, linearmodel, G, calibration, policybasis)
    
    maxregretobj = @closure threshold -> regret(weights, threshold, H̃, valuefunction, τ, Δt, (linearmodel, feedback), G, calibration, policybasis; problem = problem)
    
    r, _ = gss(maxregretobj, H̃.lb[4], H̃.ub[4]; tol=1e-2)
    
    return r
end


## Solve min max regret
minmaxregretobj = OptimizationFunction(maxregret);

optparameters = H̃, valuefunction, τ, Δt, (linearmodel, feedback), G, calibration, policybasis;

minmaxregretprob = OptimizationProblem(minmaxregretobj, weights, optparameters);

sol = solve(minmaxregretprob, COBYLA(; rhoend=1e-4));