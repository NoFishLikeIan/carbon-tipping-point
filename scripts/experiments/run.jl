using Pkg
Pkg.resolve(); Pkg.instantiate();

using JLD2
using Model, Grid
using DifferentialEquations
using FastClosures
using Dates: now
using Interpolations: Extrapolation
using UnPack: @unpack

include("arguments.jl") # Import argument parser
include("../utils/saving.jl")
include("../utils/simulating.jl")

parsedargs = ArgParse.parse_args(argtable)

@unpack simulationpath, experimentpath, datapath, verbose, trajectories = parsedargs

inputfile = joinpath(datapath, simulationpath)
if verbose ≥ 1
    println("$(now()):", "Using input file $inputfile"); flush(stdout)
end
if !isfile(inputfile)
    error("File $inputfile does not exist.")
end

result = loadtotal(inputfile);
interpolations = buildinterpolations(result);
model = last(result); timesteps = first(result);

outputfile = joinpath(datapath, experimentpath, simulationpath);
if verbose ≥ 1
    println("$(now()):", "Using output file $outputfile"); flush(stdout)
end

initialpoints = [[T₀, log(model.hogg.M₀), log(model.economy.Y₀)] for T₀ in sampletemperature(model, trajectories)];

resample = @closure (prob, id, _) -> begin
    prob = prob isa JumpProblem ? prob.prob : prob
    prob.u0[1:3] .= initialpoints[id]

    return prob
end

policies = (interpolations[:χ], interpolations[:α]);
tmin = minimum(timesteps)
tmax = maximum(timesteps) - 10.

problem = SDEProblem(F!, G!, first(initialpoints), (tmin, tmax), (model, policies))

if model isa JumpModel
    ratejump = VariableRateJump(rate, tippingopt!)
    problem = JumpProblem(problem, ratejump)
end

ensembleprob = EnsembleProblem(problem; prob_func = resample);

if verbose ≥ 1
    println("$(now()):", "Simulating $trajectories trajectories..."); flush(stdout)
end
simulation = solve(ensembleprob, SRIW1(); trajectories);
if verbose ≥ 1
    println("$(now()):", "...done!"); flush(stdout)
end

outputdir = dirname(outputfile);
if !isdir(outputdir) 
    mkpath(outputdir) 
end

JLD2.save_object(outputfile, simulation)