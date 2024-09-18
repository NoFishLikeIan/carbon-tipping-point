include("defaults.jl")

using Suppressor: @suppress
using SciMLBase: strip_solution
using DifferentialEquations

include("../utils/simulating.jl")

trajectories = 5_000;
x₀ = [hogg.T₀, log(hogg.M₀), log(Economy().Y₀)];

N = getnumber(env, "N", 51; type = Int);

optimalpath = joinpath(experimentpath, "optimal"); 
if !isdir(optimalpath) mkdir(optimalpath) end

for problemtype in keys(itpmap)
    itps = itpmap[problemtype];

    label = String(problemtype)
    println("Simulating $(label)...")

    sympath = joinpath(optimalpath, label)
    if !isdir(sympath) mkdir(sympath) end
    
    for (k, model) in enumerate(models)
        println("...model $(k)/$(length(models))...")

        interpolations = itps[model];
        policies = (interpolations[:χ], interpolations[:α]);
    
        problem = SDEProblem(F!, G!, x₀, (0., 80.), (model, policies))

        if model isa JumpModel
            ratejump = VariableRateJump(rate, tippingopt!)
            jumpprobroblem = JumpProblem(problem, ratejump)
            solution = solve(EnsembleProblem(jumpprobroblem), SRIW1(); trajectories)
        else
            solution = solve(EnsembleProblem(problem); trajectories)
        end
    
        strippedsolution = strip_solution(solution)
        filepath = joinpath(sympath, makefilename(model))
        JLD2.save_object(filepath, strippedsolution)
    end
    
    println("...saved in $(filepath) in group $label.")
end

close(experimentfile)