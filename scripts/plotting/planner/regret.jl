using Revise
using JLD2, UnPack, DataStructures
using FastClosures
using Base.Threads
using SciMLBase
using Statistics
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations, Dierckx
using StaticArrays

using Model, Grid
using Random; Random.seed!(11148705);

using Plots, PGFPlotsX, Contour
using LaTeXStrings, Printf
using Colors, ColorSchemes

# pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE,
    raw"\usepgfplotslibrary{fillbetween}",
    raw"\usetikzlibrary{patterns}",
    raw"\usepackage{siunitx}",
    raw"\DeclareSIUnit{\ppm}{p.p.m.}",
    raw"\DeclareSIUnit{\CO}{\,CO_2e}",
    raw"\DeclareSIUnit{\output}{trillion US\mathdollar / year}",
    raw"\DeclareSIUnit{\shortoutput}{tr US\mathdollar / y}",
)

includet("../../../src/valuefunction.jl")
includet("../../../src/extend/model.jl")
includet("../../../src/extend/grid.jl")
includet("../../../src/extend/valuefunction.jl")
includet("../utils.jl")
includet("../../utils/saving.jl")
includet("../../utils/simulating.jl")

damagetype = BurkeHsiangMiguel;
withnegative = true
abatementtype = withnegative ? "negative" : "constrained"
DATAPATH = "data/simulation-dense"; @assert isdir(DATAPATH)

SAVEFIG = true;
PLOTPATH = "../job-market-paper/submission/plots"
plotpath = joinpath(PLOTPATH, abatementtype)
if !isdir(plotpath) mkpath(plotpath) end

horizon = 80.
tspan = (0., horizon)

begin # Read available files
    simulationfiles = listfiles(DATAPATH)
    nfiles = length(simulationfiles)
    G = simulationfiles |> first |> loadproblem |> last
    
    modelfiles = String[]
    for (i, filepath) in enumerate(simulationfiles)
        print("Reading $i / $(length(simulationfiles))\r")
        model, _ = loadproblem(filepath)
        abatementdir = splitpath(filepath)[end - 1]

        if (model.economy.damages isa damagetype) && (abatementdir == abatementtype)
            push!(modelfiles, filepath)
        end
    end
end;

begin # Import available files
    models = IAM[]
    valuefunctions = Dict{IAM, OrderedDict{Float64, ValueFunction}}()
    interpolations = Dict{IAM, NTuple{2, Interpolations.Extrapolation}}()
    
    for (i, filepath) = enumerate(modelfiles)
        print("Loading $i / $(length(modelfiles))\r")
        values, model, Gᵢ = loadtotal(filepath; tspan=(0, 2horizon))
        interpolations[model] = buildinterpolations(values, Gᵢ);
        valuefunctions[model] = values;
        push!(models, model)
    end

    sort!(by = m -> m.climate, models)
end

begin
    calibrationpath = "data/calibration"

    # Load economic calibration
    abatementpath = joinpath(calibrationpath, "abatement.jld2")
    @assert isfile(abatementpath) "Abatement calibration file not found at $abatementpath"
    abatementfile = jldopen(abatementpath, "r+")
    @unpack abatement = abatementfile
    close(abatementfile)

    investments = Investment()
    damages = Kalkuhl()
    economy = Economy(investments = investments, damages = damages, abatement = abatement)

    # Load climate claibration
    climatepath = joinpath(calibrationpath, "climate.jld2")
    @assert isfile(climatepath) "Climate calibration file not found at $climatepath"
    climatefile = jldopen(climatepath, "r+")
    @unpack calibration, hogg, feedbacklower, feedback, feedbackhigher, decay = climatefile
    close(climatefile)
end

begin # Plot estetics
    extremamodels = (models[1], models[end - 1])
    PALETTE = colorschemes[:grays]
    colors = get(PALETTE, range(0, 0.6; length=length(models)))

    TEMPLABEL = raw"Temperature deviations $T_t \; [\si{\degree\Celsius}]$"
    LINE_WIDTH = 2.5

    Tspace, mspace = G.ranges
    temperatureticks = makedeviationtickz(Tspace[1], Tspace[end]; step=1, digits=2)

    yearlytime = range(tspan[1], tspan[2]; step=1.)
end;

function solvecorrectedpolicy(truemodel, initialmodel, ΔTᵈ, interpolations; tspan = (0., 80.), trajectories = 1_000)
    α₀ = interpolations[initialmodel][2]
    α = interpolations[truemodel][2];

    discovery = @closure (u, _, _) -> begin # Assumes T₀ < Tᶜ + ΔTᵈ
        discoverytemperature = truemodel.climate.feedback.Tᶜ + ΔTᵈ
        return discoverytemperature - u[1]
    end

    updatepolicy! = @closure integrator -> begin
        model, calibration, _ = integrator.p
        integrator.p = (model, calibration, α)
    end

    callback = ContinuousCallback(discovery, updatepolicy!)
    initialparameters = (truemodel, calibration, α₀);

    T₀ = truemodel.climate.hogg.T₀
    m₀ = log(truemodel.climate.hogg.M₀ / truemodel.climate.hogg.Mᵖ)
    u₀ = MVector(T₀, m₀, 0., 0., 0., 0.)

    prob = SDEProblem(F!, noise!, u₀, tspan, initialparameters)
    ensembleprob = EnsembleProblem(prob)

    return solve(ensembleprob, SKenCarp(); callback, trajectories)
end

begin
    fig = plot(xlabel = "Year", ylabel = "Carbon concentration")
    for ΔTᵈ in -1:3
        sol = solvecorrectedpolicy(extremamodels[1], extremamodels[2], ΔTᵈ, interpolations)
        plot!(fig, sol; idxs = 2, label = ΔTᵈ, linewidth = 2.5)
    end

    fig
end