using PGFPlotsX
using Plots
using Roots
using LaTeXStrings
using DotEnv, JLD2
using FastClosures

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
    PLOT_HORIZON = 60.
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

    for threshold in [1.5, 1.6, 1.7, 1.8, 2., 2.5]
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
    
    TEMPLABEL = L"Temperature deviations $T_t$"
    LINE_WIDTH = 2.5

    ΔTmin = hogg.T₀ - hogg.Tᵖ
    ΔTmax = 6.

    ΔTspace = range(ΔTmin, ΔTmax; length = 201);
    Tspace = ΔTspace .+ hogg.Tᵖ
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

    for (threshold, itp) in interpolations
        rowmodel = rowmodels[threshold]

        oecdpolicies = (itp[oecdmodel][:χ], itp[oecdmodel][:α]);
        rowpolicies = (itp[rowmodel][:χ], itp[rowmodel][:α]);

        policies = (oecdpolicies, rowpolicies);
        models = (oecdmodel, rowmodel);

        parameters = (models, policies, calibration);

        problem = SDEProblem(Fgame!, Ggame!, X₀, (0., PLOT_HORIZON), parameters)

        ensembleprob = EnsembleProblem(problem)

        simulation = solve(ensembleprob; trajectories = TRAJECTORIES);
        println("Done with simulation of $threshold")

        simulations[threshold] = simulation
    end
end

# Expolration
using DifferentialEquations.EnsembleAnalysis

figthresholds = [1.8, 2.5, :benchmark];
colors = Dict(figthresholds .=> [:darkred, :darkorange, :darkgreen]);
countrylabels = ["OECD", "RoW"]

qs = 0.1:0.2:0.9;
medianidx = findfirst(q -> q == 0.5, qs);

timesteps = 0:0.5:PLOT_HORIZON;
decades = 0:10:Int(PLOT_HORIZON);

# Extract quantiles
quantilesdict = Dict{Union{Float64, Symbol}, DiffEqArray}();
for threshold in figthresholds
    quantilesdict[threshold] = timeseries_point_quantile(simulations[threshold], qs, timesteps)
end

begin # State variables
    ylabels = ["\$T_{1, t}\$", "\$T_{2, t}\$", "\$M_t\$","\$Y_{1,t}\$", "\$Y_{2, t}\$"]

    defaultfig = plot(xlabel = "Year", xticks = (decades, decades .+ BASELINE_YEAR))

    statefigures = Plots.Plot{Plots.GRBackend}[]
    for idx in (1, 2, 4, 5, 3)
        idxfig = deepcopy(defaultfig)

        ylabel!(idxfig, ylabels[idx])

        if idx ≤ 2
            ylims!(idxfig, extrema(temperatureticks[1]))
            yticks!(idxfig, temperatureticks)
        end
        
        for threshold in figthresholds # Extract simulation
            quantiles = quantilesdict[threshold]

            series = getindex.(quantiles.u, idx)
            median = getindex.(series, medianidx)
            
            color = colors[threshold]

            plot!(idxfig, timesteps, median; color, linewidth = LINE_WIDTH, label = "$threshold")

            qdx = medianidx
            for dxstep in 1:(length(qs) ÷ 2)
                fillalpha = 0.1 + 0.4 * (dxstep / length(qs))

                plot!(idxfig, timesteps, getindex.(series, qdx + dxstep); fillrange = getindex.(series, qdx - dxstep), color, linewidth = 0, label = false, fillalpha)
            end

            decadequantiles = timeseries_point_median(simulations[threshold], decades)

            scatter!(idxfig, decades, getindex.(decadequantiles.u, idx); color, label = false, markerstrokewidth = 0.)
        end;

        push!(statefigures, idxfig)
    end

    l = @layout [° °; ° °; °]

    statefig = plot(statefigures...; layout = l, size = 350 .* (2√2, 4), legend = :topleft, margins = 10Plots.mm)
end

# Extract control quantiles
poldict = Dict{Union{Float64, Symbol}, Dict{Symbol, Matrix{Float64}}}();
for threshold in figthresholds
    itp = interpolations[threshold]

    ε₁ = @closure (T₁, T₂, m, y₁, y₂, t) -> begin
        α = itp[oecdmodel][:α](T₁, m, t)

        return ε(t, exp(m), α, oecdmodel, regionalcalibration, 1)
    end

    ε₂ = @closure (T₁, T₂, m, y₁, y₂, t) -> begin
        rowmodel = rowmodels[threshold]
        α = itp[rowmodel][:α](T₂, m, t)

        return ε(t, exp(m), α, rowmodel, regionalcalibration, 2)
    end

    gap = @closure (T₁, T₂, m, y₁, y₂, t) -> begin
        α₁ = itp[oecdmodel][:α](T₁, m, t)

        rowmodel = rowmodels[threshold]
        α₂ = itp[rowmodel][:α](T₂, m, t)

        return γ(t, regionalcalibration.calibration) - α₁ - α₂
    end
    
    A₁ = computeonsim(simulations[threshold], ε₁, timesteps);
    A₂ = computeonsim(simulations[threshold], ε₂, timesteps);
    G = computeonsim(simulations[threshold], gap, timesteps);

    polquantiles = Dict{Symbol, Matrix{Float64}}()
    polquantiles[:oecd] = Array{Float64}(undef, length(axes(A₁, 1)), length(qs))
    polquantiles[:row] = Array{Float64}(undef, length(axes(A₂, 1)), length(qs))
    polquantiles[:gap] = Array{Float64}(undef, length(axes(G, 1)), length(qs))

    for tdx in axes(A₁, 1)
        polquantiles[:oecd][tdx, :] .= Statistics.quantile(A₁[tdx, :], qs)
        polquantiles[:row][tdx, :] .= Statistics.quantile(A₂[tdx, :], qs)
        polquantiles[:gap][tdx, :] .= Statistics.quantile(G[tdx, :], qs)
    end

    poldict[threshold] = polquantiles
end

begin # Control
    policyfigures = Plots.Plot{Plots.GRBackend}[]
    
    for region in [:oecd, :row]
        idxfig = deepcopy(defaultfig)

        ylabel!(idxfig, L"\varepsilon_t")
        title!(idxfig, region == :oecd ? "OECD" : "ROW")

        for threshold in figthresholds
            quantiles = poldict[threshold][region]

            median = quantiles[:, medianidx]

            color = colors[threshold]

            plot!(idxfig, timesteps, median; color, linewidth = LINE_WIDTH, label = "$threshold")

            qdx = medianidx
            for dxstep in 1:(length(qs) ÷ 2)
                fillalpha = 0.1 + 0.3 * (dxstep / length(qs))

                plot!(idxfig, timesteps, quantiles[:, qdx + dxstep]; fillrange = quantiles[:, qdx - dxstep], color, linewidth = 0, label = false, fillalpha)
            end
        end;

        push!(policyfigures, idxfig)
    end

    growthfig = deepcopy(defaultfig)

    ylabel!(growthfig, L"\gamma_t - \alpha_{1,t} - \alpha_{2,t}")

    for threshold in figthresholds
        quantiles = poldict[threshold][:gap]

        median = quantiles[:, medianidx]

        color = colors[threshold]

        plot!(growthfig, timesteps, median; color, linewidth = LINE_WIDTH, label = "$threshold")

        qdx = medianidx
        for dxstep in 1:(length(qs) ÷ 2)
            fillalpha = 0.1 + 0.3 * (dxstep / length(qs))

            plot!(growthfig, timesteps, quantiles[:, qdx + dxstep]; fillrange = quantiles[:, qdx - dxstep], color, linewidth = 0, label = false, fillalpha)
        end
    end;

    l = @layout [° °; °]

    policyfig = plot(policyfigures..., growthfig; layout = l, size = 350 .* (2√2, 2), margins = 10Plots.mm)
end