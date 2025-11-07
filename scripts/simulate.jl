using Pkg
Pkg.resolve(); Pkg.instantiate();

using Base.Threads
using UnPack: @unpack
using Dates: now

include("arguments.jl") # Import argument parser

parsedargs = ArgParse.parse_args(simulateargtable)

@unpack datapath, simulationdir, calibrationpath = parsedargs # File system parameters
@unpack verbose = parsedargs # IO parameters
@unpack threshold, discovery, horizon, trajectories = parsedargs # Simulation parameters

if (verbose ≥ 1)
    println("$(now()): ", "Running with $(nthreads()) threads..."); flush(stdout)
end

# Begin script
using Model, Grid
using FastClosures
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations, Dierckx
using StaticArrays

using JLD2, DataStructures
using Printf, Dates

include("../src/valuefunction.jl")
include("../src/extend/model.jl")
include("../src/extend/grid.jl")
include("../src/extend/valuefunction.jl")

include("utils/saving.jl")
include("utils/simulating.jl")

simulationpath = joinpath(datapath, simulationdir)
@assert ispath(simulationpath) "The specified datapath does not exist: $simulationpath"

# Load climate claibration
climatepath = joinpath(datapath, calibrationpath, "climate.jld2")
@assert isfile(climatepath) "Climate calibration file not found at $climatepath"
climatefile = jldopen(climatepath, "r+")
@unpack calibration  = climatefile
close(climatefile)

# Load linear model used as incorrect
linearsolpath = joinpath(simulationpath, "linear", "growth", "logseparable", "negative", "Linear_burke_RRA10,0,jld2")
@assert ispath(linearsolpath) "The linear simulation path does not exist: $linearsolpath"

linearvalues, linearmodel, G = loadtotal(linearsolpath; tspan=(0, 1.05horizon));
_, αlinear = buildinterpolations(linearvalues, G);

# Load true threshold model
thresholdkey = replace("T$(Printf.format(Printf.Format("%.1f"), threshold))", "." => ",")
thresholdsolfile = "$(thresholdkey)_burke_RRA10,0,jld2"
thresholdsolpath = joinpath(simulationpath, "tipping", "growth", "logseparable", "negative", thresholdsolfile)

@assert ispath(thresholdsolpath) "The specified simulation file does not exist: $thresholdsolpath"

thresholdvalues, thresholdmodel, G = loadtotal(thresholdsolpath; tspan=(0, 1.05horizon));
_, αtipping = buildinterpolations(thresholdvalues, G);

temperaturetodiscovery = @closure (u, _, _) -> begin
    discoverytemperature = thresholdmodel.climate.feedback.Tᶜ + discovery
    return discoverytemperature - u[1]
end

updatepolicy! = @closure integrator -> begin
    thresholdmodel, calibration, _ = integrator.p
    integrator.p = (thresholdmodel, calibration, αtipping)
end

T₀ = thresholdmodel.climate.hogg.T₀
m₀ = log(thresholdmodel.climate.hogg.M₀ / thresholdmodel.climate.hogg.Mᵖ)
u₀ = MVector(T₀, m₀, 0., 0., 0., 0.)

initialparameters = (thresholdmodel, calibration, αlinear)
prob = SDEProblem(F!, noise!, u₀, (0., horizon), initialparameters)
ensembleprob = EnsembleProblem(prob)

callback = ContinuousCallback(temperaturetodiscovery, updatepolicy!);

if (verbose ≥ 1) println("$(now()): ", "Starting simulation..."); flush(stdout) end
sol = solve(ensembleprob, ImplicitEM(); callback, trajectories)


quantiles = EnsembleAnalysis.timeseries_point_quantile(sol, (0.01, 0.1, 0.5, 0.9, 0.99), 0:0.1:80)

outpath = joinpath("simulations", simulationdir)
if !ispath(outpath) mkpath(outpath) end
thresholdkey = replace("T$(Printf.format(Printf.Format("%.1f"), threshold))", "." => ",")
discoverykey = replace("D$(Printf.format(Printf.Format("%.1f"), discovery))", "." => ",")
outfile = joinpath(outpath, "$(thresholdkey)_$(discoverykey).jld2")

if (verbose ≥ 1) println("$(now()): ", "Saving in ", outfile); flush(stdout) end
JLD2.save_object(outfile, quantiles)