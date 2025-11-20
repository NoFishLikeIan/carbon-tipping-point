using Revise
using JLD2, UnPack, DataStructures
using FastClosures
using Base.Threads
using SciMLBase
using Statistics
using SciMLBase, DifferentialEquations, DiffEqBase
using Interpolations, ForwardDiff, Dierckx
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

horizon = 100.
tspan = (0., horizon)

begin # Read available files
    simulationfiles = listfiles(DATAPATH)
    nfiles = length(simulationfiles)
    G = simulationfiles |> first |> loadproblem |> last
    
    maximumthreshold = Inf
    modelfiles = String[]
    for (i, filepath) in enumerate(simulationfiles)
        print("Reading $i / $(length(simulationfiles))\r")
        model, _ = loadproblem(filepath)
        abatementdir = splitpath(filepath)[end - 1]

        isdamage = model.economy.damages isa damagetype
        isabatement = (abatementdir == abatementtype)
        isthreshold = model.climate isa LinearClimate || model.climate.feedback.Tᶜ ≤ maximumthreshold

        if isdamage && isabatement && isthreshold
            push!(modelfiles, filepath)
        end
    end

    println("$(length(modelfiles)) models detected.")
end;

begin # Import available files
    models = IAM[]
    valuefunctions = Dict{IAM, OrderedDict{Float64, ValueFunction}}()
    interpolations = Dict{IAM, NTuple{2, Interpolations.Extrapolation}}()
    
    for (i, filepath) = enumerate(modelfiles)
        print("Loading $i / $(length(modelfiles))\r")
        values, model, G = loadtotal(filepath; tspan=(0, 1.01horizon))
        interpolations[model] = buildinterpolations(values, G);
        valuefunctions[model] = values;
        push!(models, model)
    end

    sort!(by = m -> m.climate, models, rev = true)
end;

begin # Load calibration
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
    extremamodels = (models[1], models[end])
    extremalabels = ("Linear", "Tipping")
    PALETTE = colorschemes[:grays]
    colors = reverse(get(PALETTE, range(0, 0.6; length=length(extremamodels))))

    TEMPLABEL = raw"Temperature deviations $T_t \; [\si{\degree\Celsius}]$"
    LINE_WIDTH = 2.5

    Tspace, mspace = G.ranges

    yearlytime = range(tspan[1], tspan[2]; step=1.)
    T₀ = hogg.T₀
    m₀ = log(hogg.M₀ / hogg.Mᵖ)
    x₀ = Point(T₀, m₀)

    temperatureticks = makedeviationtickz(Tspace[1], Tspace[end]; step=1, digits=2)

    X₀ = SVector(T₀, m₀, 0.)
    u₀ = SVector(X₀..., 0., 0., 0.) # Introduce three 0s for costs
end;

begin # SCC
    tippingmodels = filter(m -> m.climate isa TippingClimate, models)
    thresholds = Float64[]
    sccs = Float64[]
    scclinear = NaN

    for model in models
        Hitp, _ = interpolations[model]
        m₀ = log(model.climate.hogg.M₀ / model.climate.hogg.Mᵖ)
        ∂Hₘ = ForwardDiff.derivative(m -> Hitp(model.climate.hogg.T₀, m, 0.), m₀)
        s = scc(∂Hₘ, model.economy.Y₀, model.climate.hogg.M₀, model)
        
        if model.climate isa TippingClimate
            push!(sccs, s)
            push!(thresholds, model.climate.feedback.Tᶜ)
        else
            scclinear = s
        end
    end

end

begin
    sccfig = @pgf Axis({ xlabel = L"Critical threshold $T^c$ [\si{\degree}]", ylabel = L"Social cost of carbon $[\si{US\mathdollar / tCe}]$", grid = "both", xmin = minimum(thresholds), xmax = maximum(thresholds) })

    curve = @pgf Plot({ color = colors[2], line_width = LINE_WIDTH }, Coordinates(thresholds, sccs))
    push!(sccfig, curve, LegendEntry(L"\mathrm{SCC}^{T^c}_0"))

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "scc.tikz"), sccfig; include_preamble=true)
    end

    sccfig
end

begin # Optimal SCC paths
    slicethresholds = [2.0]
    sccmodels = filter(m -> m.climate isa LinearClimate || (m.climate isa TippingClimate && m.climate.feedback.Tᶜ ∈ slicethresholds), models)
    m₀ = log(hogg.M₀ / hogg.Mᵖ)
    X₀ = SVector(hogg.T₀, m₀, 0.)
    trajectories = 10_000
    savestep = 0.5

    sccquantiles = Dict{IAM, Matrix{Float64}}()
    for (n, model) in enumerate(sccmodels)
        println("Solving model $n / $(length(sccmodels))")
        Hitp, αitp = interpolations[model]
        
        simulationproblem = SDEProblem(F, noise, X₀, (0, horizon), (model, calibration, αitp))

        sccfromstate = (u, t, integrator) -> begin
            model, calibration, _ = integrator.p
            T, m, y = u
            Y = exp(y) * model.economy.Y₀
            M = exp(m) * model.climate.hogg.Mᵖ
            ∂Hₘ = ForwardDiff.derivative(m′ -> Hitp(T, m′, t), m)
            return scc(∂Hₘ, Y, M, model)
        end

        savedvalues = SavedValues(Float64, Float64)
        savecallback = SavingCallback(sccfromstate, savedvalues; saveat=savestep)

        paths = Matrix{Float64}(undef, length(0:savestep:horizon), trajectories)

        for i in 1:trajectories
            if i % (trajectories ÷ 1000) == 0
                @printf("Progress %.1f%%\r", 100i / trajectories)
            end
            solve(simulationproblem; callback = savecallback)
            paths[:, i] .= savedvalues.saveval
        end

        quantiles = Matrix{Float64}(undef, size(paths, 1), 3)
        for i in axes(paths, 1), (j, q) in enumerate((0.05, 0.5, 0.9))
            v = @view paths[i, :]
            quantiles[i, j] = quantile(v, q)
        end

        sccquantiles[model] = quantiles
    end
end

let 
    yearticks = 0:20:horizon
    fig = @pgf Axis({
        xlabel = L"Year",
        ylabel = L"Social cost of carbon $[\si{US\mathdollar / tCe}]$",
        grid = "both",
        xmin = 0, ymin = 0,
        xmax = horizon,
        xtick = yearticks, xticklabels = floor.(Int64, yearticks .+ 2020),
        legend_pos = "north west"
    })

    timegrid = 0:savestep:horizon

    for (i, model) in enumerate(sccmodels)
        quantiles = sccquantiles[model]

        for col in eachcol(quantiles) smooth!(col, 10) end

        label = model.climate isa LinearClimate ? "Linear" : "Tipping"
        color = colors[i]
        
        # Median line
        median_coords = Coordinates(timegrid, quantiles[:, 2])
        push!(fig, @pgf Plot({color = color, line_width = LINE_WIDTH}, median_coords))
        push!(fig, LegendEntry(label))
        
        # Shaded region for 5th-95th percentile
        lower_coords = Coordinates(timegrid, quantiles[:, 1])
        upper_coords = Coordinates(timegrid, quantiles[:, 3])
        
        push!(fig, @pgf Plot({color = color, opacity = 0.15, fill = color, forget_plot = true}, 
            Table([timegrid; reverse(timegrid)], [quantiles[:, 1]; reverse(quantiles[:, 3])])))
    end

    if SAVEFIG
        PGFPlotsX.save(joinpath(plotpath, "scc-paths.tikz"), fig; include_preamble=true)
    end

    fig
end