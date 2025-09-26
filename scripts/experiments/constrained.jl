using Interpolations: Extrapolation
using Dates
using JLD2
using Model, Grid
using SciMLBase, DifferentialEquations
using FastClosures

include("../utils/saving.jl")
include("../utils/simulating.jl")

withnegative = true;
trajectories = 5_000;
datapath = "data/simulation-small";

println("Loading interpolations...")
begin
    filepaths = joinpath(datapath, withnegative ? "negative" : "constrained")

    simulationfiles = listfiles(filepaths)

    interpolations = Dict{Symbol, Extrapolation}[];
    models = AbstractModel[];

    for filepath in simulationfiles
        result = loadtotal(filepath)
        itp = buildinterpolations(result)
        model = last(result)

        push!(interpolations, itp)
        push!(models, model)
    end
end;

experimentdir = joinpath(filepaths, "experiments")
if !isdir(experimentdir)
    mkdir(experimentdir)
end

experimentpath = joinpath(experimentdir, withnegative ? "negative.jld2" : "constrained.jld2");
if isdir(experimentpath)
    unix = round(Int, time())
    newfilepath = joinpath(experimentdir, withnegative ? "negative_$unix.jld2" : "constrained_$unix.jld2")

    mv(experimentpath, newfilepath)
    rm(experimentpath; recursive = true)

    @warn "Moved $experimentpath -> $newfilepath"
end

timesteps = range(0., 80.; step = 0.25);
println("Running experiments...")
jldopen(experimentpath, "w") do file
    for k in axes(models, 1)

        model = models[k]
        itp = interpolations[k]

        policies = (itp[:χ], itp[:α]);
        parameters = (model, policies);

        m₀ = log(model.climate.hogg.M₀)
        y₀ = log(model.economy.Y₀)
        initialpoints = [[T₀, m₀, y₀] for T₀ in sampletemperature(model, trajectories)];

        resample = @closure (prob, id, _) -> begin
            prob = prob isa JumpProblem ? prob.prob : prob

            prob.u0[1:3] .= initialpoints[id]
            return prob
        end

        problem = SDEProblem(F!, G!, Vector{Float64}(undef, 3), extrema(timesteps), (model, policies))

        if model isa JumpModel
            ratejump = VariableRateJump(rate, tippingopt!)
            problem = JumpProblem(problem, ratejump)
        end

        ensembleproblem = EnsembleProblem(problem, prob_func = resample)
        simulation = solve(ensembleproblem, SRIW1(); trajectories = trajectories)

        T = length(timesteps)
        data = Array{Float64, 3}(undef, trajectories, T, 3)

        for (i, t) in enumerate(timesteps)
            points = simulation(t)

            for (k, point) in enumerate(points)
                data[k, i, :] .= point[1:3]
            end
        end

        group = JLD2.Group(file, "$k")
        group["model"] = model
        group["data"] = data

        print("Saved simulation $k\r")
    end
    println("\nDone! Closing $experimentpath.")
end
