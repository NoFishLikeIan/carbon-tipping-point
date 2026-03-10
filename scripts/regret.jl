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
using NLopt

using LinearSolve, LinearAlgebra

using JLD2, UnPack
using Dates, Printf

includet("../src/valuefunction.jl")
includet("../src/extend/model.jl")
includet("../src/extend/grid.jl")
includet("../src/extend/valuefunction.jl")
includet("../src/regret.jl")
includet("utils/saving.jl")
includet("utils/simulating.jl")
includet("utils/loading.jl")
includet("plotting/utils.jl")
includet("utils/simulating.jl")
includet("utils/approximate.jl")
includet("markov/chain.jl")
includet("markov/finitedifference.jl")
includet("regret/chain.jl")
includet("regret/finitedifference.jl")

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
Δt⁻¹ = 8.
Δt = 1 / Δt⁻¹
τ = 100.

preferences = LogSeparable()
decay = ConstantDecay(0.)
climate = LinearClimate(hogg, decay)

preferences = LogSeparable()
linearmodel = IAM(climate, economy, preferences)

## Construct policy simplex
simpath = "data/simulation";
paths = loadsimulationpaths(simpath; exclude = ["terminal", "linear"])
K = 5;
_, G = loadproblem(paths[2.0])

filteredpaths = basispolicypaths(paths, K, G) # Indices of the basis
policybasis = SimplexPolicies(filteredpaths);

## Construct Chebyshev reresentation of full information
G = coarse(G, (2, 2))
values = OrderedDict(k => loadtotal(p) for (k, p) in paths)

order = (20, 20, 10, 5)
H̃ = chebyshevrepresentation(values, order);

## Compute regret
valuefunction = ValueFunction(τ, climate, G, calibration)

weights = MVector{K}(rand(K));
weights ./= sum(weights)

function regret(weights, threshold, optparameters)
    H̃, valuefunction, τ, Δt, (linearmodel, feedback), G, calibration, policybasis = optparameters

    regret(weights, threshold, H̃, valuefunction, τ, Δt, (linearmodel, feedback), G, calibration, policybasis)
end
optparameters = H̃, valuefunction, τ, Δt, (linearmodel, feedback), G, calibration, policybasis;

regret(weights, 2., optparameters)

## Setup epigraph method # min{t | t, w ≥ 0} s.t. t ≥ r(w, Tᶜ) ∀ Tᶜ ∈ thresholds
function objective(u, _, optparameters)
    u[1]
end
function feasibility(u, _, threshold, optparameters)
    policybasis = last(optparameters)
    
    K = size(policybasis)
    w = @view u[2:(K + 1)]
    weights = SVector{K}(w ./ sum(w))
    return regret(weights, threshold, optparameters) - u[1]
end

## Solve 
opt = NLopt.Opt(:LN_COBYLA, K + 1)
NLopt.min_objective!(opt, Base.Fix{3}(objective, optparameters))
NLopt.xtol_rel!(opt, 1e-3)
NLopt.lower_bounds!(opt, zeros(K + 1))
NLopt.upper_bounds!(opt, [Inf, ones(K)...])

optfeasibility = Base.Fix{4}(feasibility, optparameters)
for threshold in policybasis.thresholds
    NLopt.inequality_constraint!(opt, Base.Fix{3}(optfeasibility, threshold), 1e-6)
end

u₀ = [1., weights...]
y, u, reason = NLopt.optimize(opt, u₀)
weights = SVector{K}(u[2:end] ./ sum(u[2:end]))

r, Tᶜ = gss(threshold -> regret(weights, threshold, optparameters), 2., 4., tol = 1e-3)

## Save result
using JLD2

JLD2.@save "data/regret/policy.jld2" weights policybasis r Tᶜ