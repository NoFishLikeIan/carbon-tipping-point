using JLD2
using Model, Grid
using Random

using Interpolations: Extrapolation
using SciMLBase
using DifferentialEquations
using FastClosures
using Dates

includet("../utils/saving.jl")
includet("../utils/simulating.jl")

ALLOWNEGATIVE = false;
SEED = 1148705;
rng = MersenneTwister(SEED);
trajectories = 10_000;
datapath = "data/simulation-small";

begin
    filepaths = joinpath(datapath, ALLOWNEGATIVE ? "negative" : "constrained")

    simulationfiles = listfiles(filepaths)

    itpmap = Dict{AbstractModel, Dict{Symbol, Extrapolation}}();
    models = AbstractModel[];

    for filepath in simulationfiles
        result = loadtotal(filepath)
        interpolations = buildinterpolations(result)
        model = last(result)

        itpmap[model] = interpolations
        push!(models, model)
    end
end;

experimentdir = joinpath(filepaths, "experiments")
if !isdir(experimentdir)
    mkdir(experimentdir)
end

experimentpath = joinpath(experimentdir, ALLOWNEGATIVE ? "negative.jld2" : "constrained.jld2");
if isfile(experimentpath)
    unix = round(Int, time())
    newfilepath = joinpath(experimentdir, ALLOWNEGATIVE ? "negative_$unix.jld2" : "constrained_$unix.jld2")

    mv(experimentpath, newfilepath)
end

jldopen(experimentpath, "w") do experimentfile
    for (k, model) in enumerate(models)
        println("Simulation $(k)/$(length(models))...")
        
        group = JLD2.Group(experimentfile, string(k))
    
        interpolations = itpmap[model];
        policies = (interpolations[:χ], interpolations[:α]);
        parameters = (model, policies);
    
        initialpoints = [[T₀, log(model.hogg.M₀), log(model.economy.Y₀)] for T₀ in sampletemperature(rng, model, trajectories)];
    
        resample = (prob, id, _) -> begin
            prob.u0 .= initialpoints[id]
            return prob
        end
    
        problem = SDEProblem(F!, G!, first(initialpoints), (0., 80.), (model, policies))
    
        ensembleprob, solver = if model isa JumpModel
            ratejump = VariableRateJump(rate, tippingopt!)
            problem = JumpProblem(problem, ratejump)
    
            EnsembleProblem(problem; prob_func = resample), ImplicitEM() 
        else
            EnsembleProblem(problem; prob_func = resample), SRIW1()
        end
    
        simulation = solve(ensembleprob, solver; trajectories = trajectories, seed = SEED)
    
        group["model"] = model
        group["simulation"] = SciMLBase.strip_solution(simulation)
    end
end