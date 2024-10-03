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

@unpack inputpath, outputpath, verbose, trajectories = parsedargs

if verbose ≥ 1
    println("$(now()): ", "Using input file $inputpath"); flush(stdout)
end
if !isfile(inputpath)
    error("File $inputpath does not exist.")
end

result = loadtotal(inputpath);
interpolations = buildinterpolations(result);
model = last(result); timesteps = first(result);

begin # Setup
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
        problem = JumpProblem(problem, Direct(), ratejump)
    end

    ensembleprob = EnsembleProblem(problem; prob_func = resample)
end

begin # Simulation
    if verbose ≥ 1
        println("$(now()):", "Simulating $trajectories trajectories with $(Base.Threads.nthreads()) threads."); flush(stdout)
    end
    
    solver = model isa JumpModel ? ImplicitEM() : SRIW1()

    simulation = solve(ensembleprob, SRIW1(); trajectories);
    if verbose ≥ 1
        println("$(now()):", "...done!"); flush(stdout)
    end
end

if verbose ≥ 1
    println("$(now()):", "Using output file $outputpath"); flush(stdout)
end

outputdir = dirname(outputpath);
if !isdir(outputdir) mkpath(outputdir) end

JLD2.save_object(outputpath, simulation)