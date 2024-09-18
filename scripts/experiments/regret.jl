include("defaults.jl")
using Suppressor: @suppress

using Distributed
println("Simulating with $(nprocs()) processes.")

using DifferentialEquations, SciMLBase

@everywhere include("../utils/simulating.jl")

trajectories = 10_000;
x₀ = [hogg.T₀, log(hogg.M₀), log(economy.Y₀)];

N = getnumber(env, "N", 51; type = Int);

if !isdir(experimentpath) mkdir(experimentpath) end

filepath = joinpath(experimentpath, "regret_$N.jld2")
experimentfile = jldopen(filepath, "w")

for (sym, label) in [(:constrained, "constrained"), (:negative, "negative")]
    println("Simulating $(label)...")

    group = JLD2.Group(experimentfile, label);

    itps = itpmap[:constrained];
    modelimminent, modelremote = tippingmodels;

    policiesimminent = (itps[modelimminent][:χ], itps[modelimminent][:α]);
    policiesremote = (itps[modelremote][:χ], itps[modelremote][:α]);

    initparams = (modelimminent, policiesremote); # The tipping point is imminent, yet the strategy is remote

    regretprob = SDEProblem(F!, G!, x₀, (0., 80.), initparams)
    regretensemble = EnsembleProblem(regretprob)

    function hittipping(u, t, integrator)
        model = first(integrator.p)
        Thit = model.albedo.Tᶜ + model.hogg.Tᵖ + (model.albedo.ΔT / 4)

        return Thit - u[1]
    end

    function changepolicy!(integrator)
        integrator.p = (modelimminent, policiesimminent) # Realises that the tipping point is imminent and switch strategy
    end

    callback = ContinuousCallback(hittipping, changepolicy!);

    regretsolution = solve(regretensemble; trajectories, callback);

    strippedsolution = SciMLBase.strip_solution(regretsolution)
    
    @suppress group["solution"] = strippedsolution

    println("...saved in $(filepath) in group $label.")
end

close(experimentfile)