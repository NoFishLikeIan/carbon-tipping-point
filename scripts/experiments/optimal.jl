include("defaults.jl")
using Suppressor: @suppress

using Distributed
println("Simulating with $(nprocs()) processes.")

using DifferentialEquations, SciMLBase

@everywhere include("../utils/simulating.jl")

trajectories = 10_000;
x₀ = [hogg.T₀, log(hogg.M₀), log(economy.Y₀)];

ratejump = VariableRateJump(rate, tippingopt!);

N = getnumber(env, "N", 51; type = Int);

if !isdir(experimentpath) mkdir(experimentpath) end

filepath = joinpath(experimentpath, "experiment_$N.jld2")
experimentfile = jldopen(filepath, "w")

for (sym, label) in [(:constrained, "constrained"), (:negative, "negative")]
    println("Simulating $(label)...")

    itps = itpmap[sym];

    group = JLD2.Group(experimentfile, label);
    
    for (k, model) in enumerate(models)
        println("...model $(k)/$(length(models))...")

        interpolations = itps[model];
        policies = (interpolations[:χ], interpolations[:α]);
    
        problem = SDEProblem(F!, G!, x₀, (0., 80.), (model, policies))

        if model isa JumpModel
            jumpprobroblem = JumpProblem(problem, ratejump)
            solution = solve(EnsembleProblem(jumpprobroblem), SRIW1(), EnsembleDistributed(); trajectories)
        else
            solution = solve(EnsembleProblem(problem), EnsembleDistributed(); trajectories)
        end
    
        strippedsolution = SciMLBase.strip_solution(solution)
    
        @suppress group[modellabels[model]] = strippedsolution
    end
    
    println("...saved in $(filepath) in group $label.")
end

close(experimentfile)