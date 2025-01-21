using PGFPlotsX
using Plots
using Roots
using LaTeXStrings
using DifferentialEquations
using DotEnv, JLD2

using Colors, ColorSchemes

using Random

using Plots, PGFPlotsX
using LaTeXStrings, Printf
using Colors, ColorSchemes

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}", raw"\usetikzlibrary{patterns}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{siunitx}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\DeclareSIUnit{\ppm}{p.p.m.}")

using Statistics
using Model, Grid
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations: Extrapolation
using Dierckx, ImageFiltering

includet("utils.jl")
includet("../utils/saving.jl")
includet("../utils/simulating.jl")

begin # Environment variables
    env = DotEnv.config(".env.game")
    plotpath = get(env, "PLOTPATH", "plots")
    datapath = get(env, "DATAPATH", "data")
    simulationpath = get(env, "SIMULATIONPATH", "simulations")

    BASELINE_YEAR = 2020
    PLOT_HORIZON = 80.
    SAVEFIG = false

    calibration = load_object(joinpath(datapath, "calibration.jld2"))
    regionalcalibration = load_object(joinpath(datapath, "regionalcalibration.jld2"))
end;

begin # Models definition
    # -- Climate
    hogg = Hogg()
    # -- Economy and Preferences
    preferences = EpsteinZin();
    oecdeconomy, roweconomy = RegionalEconomies()
    damages = GrowthDamages()

    oecdmodel = LinearModel(hogg, preferences, damages, oecdeconomy)

    rowmodels = Dict{Union{Float64, Symbol}, AbstractModel}()

    rowmodels[:benchmark] =  LinearModel(hogg, preferences, damages, roweconomy);

    for threshold in [1.5, 2., 2.5]
        rowmodels[threshold] = TippingModel(Albedo(threshold), hogg, preferences, damages, roweconomy);
    end
end

begin # Load simulations and build interpolations
    interpolations = Dict{
        Union{Float64, Symbol}, 
        Dict{
            AbstractModel, 
            Dict{Symbol, Extrapolation}
        }}();

    for (threshold, rowmodel) in rowmodels
        result = loadgame(AbstractModel[oecdmodel, rowmodel]; outdir = simulationpath)

        interpolations[threshold] = buildinterpolations(result)
    end
end

begin # Plot estetics
    PALETTE = colorschemes[:grays];
    colors = get(PALETTE, [0., 0.5]);
    PLOT_HORIZON = 80.
    
    TEMPLABEL = L"Temperature deviations $T_t$"
    LINE_WIDTH = 2.5

    ΔTmin = Hogg().T₀ - Hogg().Tᵖ
    ΔTmax = 3. 

    ΔTspace = range(ΔTmin, ΔTmax; length = 201);
    Tspace = ΔTspace .+ Hogg().Tᵖ
    Tmin, Tmax = extrema(Tspace)

    mspace = range(mstable(Tmin, oecdmodel), mstable(Tmax, oecdmodel); length = length(ΔTspace))

    yearlytime = range(0., PLOT_HORIZON; step = 1.)
    temperatureticks = makedeviationtickz(ΔTmin, ΔTmax, oecdmodel; step = 1, digits = 0)
end;

# -- Make simulation of optimal trajectories
begin
    TRAJECTORIES = 10_000;
    simulations = Dict{Union{Float64, Symbol}, EnsembleSolution}();

    # The initial state is given by (T₁, T₂, m, y₁, y₂)
    X₀ = [Hogg().T₀, Hogg().T₀, log(Hogg().M₀), log(oecdeconomy.Y₀), log(roweconomy.Y₀)];

    for threshold in keys(interpolations)
        itp = interpolations[threshold];

        oecdpolicies = (itp[oecdmodel][:χ], itp[oecdmodel][:α]);
        rowpolicies = (itp[rowmodels[threshold]][:χ], itp[rowmodels[threshold]][:α]);

        policies = PoliciesFunctions[oecdpolicies, rowpolicies];
        parameters = (models, policies, calibration);

        problem = SDEProblem(Fgame!, Ggame!, X₀, (0., PLOT_HORIZON), parameters)

        ensembleprob = EnsembleProblem(problem)

        simulation = solve(ensembleprob; trajectories = TRAJECTORIES);
        println("Done with simulation of $model")

        simulations[model] = simulation
    end
end