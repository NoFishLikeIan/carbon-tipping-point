includet("defaults.jl")

using Distributed
println("Simulating with $(nprocs()) processes.")

using DifferentialEquations

@everywhere include("../utils/simulating.jl")

trajectories = 10_000;
x₀ = [hogg.T₀, log(hogg.M₀), log(economy.Y₀)];

ratejump = VariableRateJump(rate, tippingopt!);

for (sym, label) in [(:constrained, "constrained"), (:negative, "negative")]
    println("Simulating $(label)...")

    itps = itpmap[:constrained];
    
    solutions = Dict{AbstractModel, EnsembleSolution}();
    
    for (k, model) in enumerate(models)
        println("...model $(k)/$(length(models))...")

        interpolations = itps[model]
        policies = (interpolations[:χ], interpolations[:α])
    
        problem = SDEProblem(F!, G!, x₀, (0., 80.), (model, policies))
    
    
        if model isa JumpModel
            jumpprobroblem = JumpProblem(problem, ratejump)
            solution = solve(EnsembleProblem(jumpprobroblem), SRIW1(), EnsembleDistributed(); trajectories)
        else
            solution = solve(EnsembleProblem(problem), EnsembleDistributed(); trajectories)
        end
    
    
        solutions[model] = solution
    end

    
    filepath = joinpath(experimentpath, "$label.jld2")
    @save filepath solutions

    println("...saved in $(filepath).")
end