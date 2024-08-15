
using Revise
using JLD2, DotEnv, CSV
using UnPack
using DataFrames, DataStructures

using FiniteDiff
using DifferentialEquations, DifferentialEquations.EnsembleAnalysis

using KernelDensity
using Interpolations
using Interpolations: Extrapolation

using Plots, Printf, PGFPlotsX, Colors, ColorSchemes
using Statistics

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")

using Model, Grid

includet("../utils/saving.jl")
includet("../utils/simulating.jl")
includet("utils.jl")

begin # Global variables
    env = DotEnv.config()
    BASELINE_YEAR = 2020
    DATAPATH = get(env, "DATAPATH", "data")
    PLOTPATH = get(env, "PLOTPATH", "plots")
    PRESENTATIONPATH = joinpath(PLOTPATH, "presentation")

    SAVEFIG = false 
    kelvintocelsius = 273.15
    LINE_WIDTH = 2.5
    SEED = 11148705
end;

begin # Construct models and grids
    thresholds = [1.5, 2.5];
	N = 51;

	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	preferences = EpsteinZin()
    damages = GrowthDamages()
    economy = Economy()
    jump = Jump()
    hogg = Hogg()

    jumpmodel = JumpModel(jump, preferences, damages, economy, Hogg(), calibration)
    
	models = AbstractModel[]
    Gs = RegularGrid[]

	for Tᶜ ∈ thresholds
	    albedo = Albedo(Tᶜ = Tᶜ)
	    model = TippingModel(albedo, preferences, damages, economy, hogg, calibration)

        G = constructdefaultgrid(N, model)

		push!(models, model)
        push!(Gs, G)
	end

    jumpmodel = JumpModel(jump, preferences, damages, economy, hogg, calibration)
    push!(models, jumpmodel)
    push!(Gs, constructdefaultgrid(N, jumpmodel))
end;

begin # Labels, colors and axis
    PALETTE = colorschemes[:grays]
    graypalette = n -> n > 1 ? 
        get(PALETTE, range(0.1, 0.8; length = n)) : 0.8

    thresholdcolor = Dict(thresholds .=> graypalette(length(thresholds)))

    TEMPLABEL = "Temperature deviations \$T_t - T^{p}\$"

    ΔTmax = 8.
    ΔTspace = range(0., ΔTmax; length = 51)
    Tspace = ΔTspace .+ Hogg().Tᵖ

    horizon = round(Int64, calibration.tspan[2])
    yearlytime = 0:1:horizon

    temperatureticks = makedeviationtickz(0., ΔTmax, first(models); step = 1, digits = 0)

    Tmin, Tmax = extrema(temperatureticks[1])

    X₀ = [hogg.T₀, log(hogg.M₀)]

    baufn = SDEFunction(Fbau!, G!)
end;

begin # Load IPCC data
    IPCCDATAPATH = joinpath(DATAPATH, "climate-data", "proj-median.csv")
    ipccproj = CSV.read(IPCCDATAPATH, DataFrame)

    getscenario(s::Int64) = filter(:Scenario => isequal("SSP$(s) - Baseline"), ipccproj)

    bauscenario = getscenario(5)
end;

# --- Optimal emissions 
results = loadtotal(models, Gs; datapath = DATAPATH);
itps = buildinterpolations(results, Gs);
modelmap = Dict(models .=> itps);

function F!(dx, x, p, t)	
	model, αitp = p
	T, m = x

    α = αitp(T, m, t)
	
	dx[1] = μ(T, m, model) / model.hogg.ϵ
	dx[2] = γ(t, model.calibration) - α
end;

function G!(Σ, x, p::Tuple{AbstractModel, Any}, t)
    G!(Σ, x, first(p), t)
end;

rate(u, p::Tuple{JumpModel, Any}, t) = rate(u, first(p), t)
function tippingaffect!(integrator)
    model = first(integrator.p)
    q = increase(integrator.u[1], model.hogg, model.jump)
    integrator.u[1] += q
end

ratejump = VariableRateJump(rate, tippingaffect!);

begin # Solve an ensemble problem for all models with the bau scenario
    sols = Dict{AbstractModel, EnsembleSolution}()
    x₀ = [hogg.T₀, log(hogg.M₀)]

    for model in models
        itp = modelmap[model]
        αitp = itp[:α]

        p = (model, αitp);

        prob = SDEProblem(F!, G!, x₀, (0., 80.), p)

        if typeof(model) <: JumpModel
            jumpprob = JumpProblem(prob, ratejump)
            ensprob = EnsembleProblem(jumpprob)
            abatedsol = solve(ensprob, SRIW1(); trajectories = 1_000)
        else
            ensprob = EnsembleProblem(prob)
            abatedsol = solve(ensprob; trajectories = 10_000)
        end

        sols[model] = abatedsol
    end
end;

function smoothedβ(path, αitp, model; w = 2)
    n = length(path)
    raw = Vector{Float64}(undef, n)
    
    for j in 1:n
        t = path.t[j]
        T, m = path.u[j]
        M = exp(m)

        α = αitp(T, m, t)

        raw[j] = Model.β(t, Model.ε(t, M, α, model), model.economy)
    end

    ma = copy(raw)

    for j in 1:n
        from = max(j - w, 1)
        to = min(j + w, n)        
        ma[j] = mean(raw[from:to])
    end
    
    return ma
end

# Constructs a Group plot, one for the path of T and one for M
begin
    simfig = @pgf GroupPlot({
        group_style = { 
            group_size = "2 by $(length(models))",
            horizontal_sep = raw"0.15\textwidth",
            vertical_sep = raw"0.1\textwidth"
        },
        width = raw"\textwidth", height = raw"\textwidth",
    });

    yearticks = 0:20:horizon

    βextrema = (0., 0.04)
    βticks = range(βextrema...; step = 0.01) |> collect
    βticklabels = [@sprintf("%.0f \\%%", 100 * y) for y in βticks]

    Textrema = hogg.Tᵖ .+ (1., 2.)
    Tticks = makedeviationtickz((Textrema .- hogg.Tᵖ)..., first(models); step = 0.5, digits = 1)

    for (k, model) in enumerate(models)
        abatedsol = sols[model]
        itp = modelmap[model]
        αitp = itp[:α]

        medianpath = timeseries_point_median(abatedsol, yearlytime)
        lowerpath = timeseries_point_quantile(abatedsol, 0.1, yearlytime)
        upperpath  = timeseries_point_quantile(abatedsol, 0.9, yearlytime)

        Tmedian = @. first(medianpath.u)
        Tlower = @. first(lowerpath.u)
        Tupper = @. first(upperpath.u)

        βmedian = smoothedβ(medianpath, αitp, model; w = 10)
        βlower = smoothedβ(lowerpath, αitp, model; w = 10)
        βupper = smoothedβ(upperpath, αitp, model; w = 10)

        clamped = clamp.(βmedian, βlower, βupper)

        # β figure
        βmedianplot = @pgf Plot({ line_width = LINE_WIDTH }, Coordinates(zip(yearlytime, clamped)))
        βlowerplot = @pgf Plot({ line_width = LINE_WIDTH, dotted, opacity = 0.5 }, Coordinates(zip(yearlytime, βlower)))
        βupperplot = @pgf Plot({ line_width = LINE_WIDTH, dotted, opacity = 0.5 }, Coordinates(zip(yearlytime, βupper)))

        βoptionfirst = @pgf k > 1 ? {} : {
            title = raw"Abatement as \% of $Y_t$",
        }

        βoptionlast = @pgf k < length(models) ?
            { 
                xticklabels = raw"\empty"
            } : {
                xtick = yearticks,
                xticklabels = BASELINE_YEAR .+ yearticks,
                xticklabel_style = { rotate = 45 }
            }

        

        ylabel = if typeof(model) <: TippingModel
            if model.albedo.Tᶜ < 2
                raw"\textit{imminent}"
            else
                raw"\textit{far}"
            end
        else
            raw"\textit{benchmark}"
        end

        @pgf push!(simfig, {
            width = raw"0.42\textwidth",
            height = raw"0.3\textwidth",
            grid = "both",
            ymin = βextrema[1], ymax = βextrema[2],
            ytick = βticks,
            xmin = 0, xmax = horizon,
            scaled_y_ticks=false,
            yticklabels = βticklabels,
            ylabel = ylabel,
            βoptionlast..., βoptionfirst...
        }, βmedianplot, βlowerplot, βupperplot)

        # T figure
        Tmedianplot = @pgf Plot({ line_width = LINE_WIDTH }, Coordinates(zip(yearlytime, Tmedian)))
        Tlowerplot = @pgf Plot({ line_width = LINE_WIDTH, dotted, opacity = 0.5 }, Coordinates(zip(yearlytime, Tlower)))
        Tupperplot = @pgf Plot({ line_width = LINE_WIDTH, dotted, opacity = 0.5 }, Coordinates(zip(yearlytime, Tupper)))

        Ttickoptions = @pgf k < length(models) ?
            { 
                xticklabels = raw"\empty"
            } : {
                xtick = yearticks,
                xticklabels = BASELINE_YEAR .+ yearticks,
                xticklabel_style = { rotate = 45 }
            }

        Ttitleoptions = @pgf k > 1 ? {} : {
                title = raw"Temperature $T_t$"
            }

        @pgf push!(simfig, {
            width = raw"0.42\textwidth",
            height = raw"0.3\textwidth",
            grid = "both",
            xmin = 0, xmax = horizon,
            ymin = Textrema[1], ymax = Textrema[2],
            ytick = Tticks[1], yticklabels = Tticks[2],
            Ttickoptions..., Ttitleoptions...
        }, Tmedianplot, Tlowerplot, Tupperplot)
    end;

    if SAVEFIG
        PGFPlotsX.save(joinpath(PLOTPATH, "simfig.tikz"), simfig; include_preamble = true) 
    end

    simfig
end

# Abatement spending
begin
    G = first(Gs)
    mspace = range(G.domains[2]...; length = size(G, 2))
    
    bmodel = last(models)
    bitp = modelmap[bmodel]
    bαitp = bitp[:α]  

    model = models[1]
    itp = modelmap[model]
    αitp = itp[:α]
    t = 0.

    function Δβ(m, T)
        M = exp(m)
        α₁ = αitp(T, m, 0.)
        α₂ = bαitp(T, m, 0.)

        Model.β(t, Model.ε(t, M, α₁, model), model.economy) -
        Model.β(t, Model.ε(t, M, α₂, model), model.economy)
    end

    surface(mspace, Tspace, Δβ; c = :coolwarm, camera = (45, 45), xflip = true)

end