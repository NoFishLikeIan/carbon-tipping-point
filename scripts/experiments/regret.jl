includet("defaults.jl")

using Distributed
println("Simulating with $(nprocs()) processes.")

using DifferentialEquations

@everywhere include("../utils/simulating.jl")

trajectories = 50_000;
x₀ = [hogg.T₀, log(hogg.M₀), log(economy.Y₀)];

for (sym, label) in [(:constrained, "constrained"), (:negative, "negative")]
    println("Simulating $(label)...")

    itps = itpmap[:constrained];

    modelimminent, modelremote = tippingmodels;
    αimminent = itps[modelimminent][:α];
    αremote = itps[modelremote][:α];

    initparams = (modelimminent, αremote); # The tipping point is imminent, yet the strategy is remote

    regretprob = SDEProblem(F!, G!, x₀, (0., 80.), initparams) |> EnsembleProblem
    
    function hittipping(u, t, integrator)
        model = first(integrator.p)
        Thit = model.albedo.Tᶜ + model.hogg.Tᵖ + (model.albedo.ΔT / 4)

        return Thit - u[1]
    end

    function changepolicy!(integrator)
        integrator.p = (modelimminent, αimminent) # Realises that the tipping point is imminent and switch strategy
    end

    callback = ContinuousCallback(hittipping, changepolicy!);

    regretsolution = solve(regretprob; trajectories, callback);
    
    filepath = joinpath(experimentpath, "regret-$label.jld2")
    @save filepath regretsolution

    println("...saved in $(filepath).")
end