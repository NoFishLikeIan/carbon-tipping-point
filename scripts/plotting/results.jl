using Revise
using JLD2, DotEnv, CSV
using UnPack
using DataFrames, DataStructures

using FiniteDiff
using DifferentialEquations, DifferentialEquations.EnsembleAnalysis

using KernelDensity
using Interpolations
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
    thresholds = [1.5, 1.8, 2., 3.4]
	N = 51;

	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	preferences = EpsteinZin()
    damages = GrowthDamages()
    economy = Economy()
    jump = Jump()
    hogg = Hogg()

    jumpmodel = JumpModel(jump, preferences, damages, economy, Hogg(), calibration)
    
	models = TippingModel[]
    Gs = RegularGrid[]

	for Tᶜ ∈ thresholds
	    albedo = Albedo(Tᶜ = Tᶜ)
	    model = TippingModel(albedo, preferences, damages, economy, hogg, calibration)

        G = constructdefaultgrid(N, model)

		push!(models, model)
        push!(Gs, G)
	end
end;

begin # Labels, colors and axis
    PALETTE = colorschemes[:grays]
    graypalette = n -> get(PALETTE, range(0.1, 0.8; length = n))

    thresholdcolor = Dict(thresholds .=> graypalette(length(thresholds)))

    TEMPLABEL = "Temperature deviations \$T_t - T^{p}\$"

    ΔTmax = 8.
    ΔTspace = range(0., ΔTmax; length = 51)
    Tspace = ΔTspace .+ Hogg().Tᵖ

    horizon = round(Int64, calibration.tspan[2])
    yearlytime = 0:1:horizon

    temperatureticks = collect.(makedeviationtickz(0., ΔTmax, first(models); step = 1, digits = 0))

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
itps = buildinterpolations(results, Gs; kx = 10, ky = 10, s = 1_000);

modelpamp = Dict(models .=> itps);

Tspan, mspan = first(Gs).domains;
Tspace = range(Tspan...; length = 1001);
mspace = range(mspan...; length = 1001);

# Policy
itpcontourf(itp; kwargs...) = itpcontourf!(plot(), itp; kwargs...)
function itpcontourf!(fig, itp; space = (mspace, Tspace), plotkwargs...)
    contour!(fig, 
        space[1], space[2], 
        (m, T) -> evaluate(itp, T, m);
        xlabel = raw"Carbon concentration, \$m\$",
        ylabel = raw"Temperature, \$T\$",
        plotkwargs...
    )
end

αitp = itps[1][0.][3];

contourf( 
        mspace, Tspace, 
        (m, T) -> Model.ε(0., exp(m), evaluate(αitp, T, m), first(models));
        cmap = reverse(cgrad(:grays)), clims = (0, 1),
        xlabel = raw"Carbon concentration, \$m\$",
        ylabel = raw"Temperature, \$T\$",
        lw = 0., 
)

# Simulation
function F!(dx, x, p::Tuple{TippingModel, Extrapolation, Extrapolation}, t)	
	model, χitp, αitp = p
	
	T, m = x
	
	χ = χitp(T, m, t)
	α = αitp(T, m, t)
	
	dx[1] = μ(T, m, model.hogg, model.albedo) / model.hogg.ϵ
	dx[2] = γ(t, model.calibration) - α
end;

function G!(Σₓ, x, p, t)
	model = first(p)
	
	Σₓ[1] = model.hogg.σₜ / model.hogg.ϵ
	Σₓ[2] = model.hogg.σₘ
end;

x₀ = [hogg.T₀, log(hogg.M₀)];
model = first(models);
p = (model, first(itps)[2], first(itps)[3]);
prob = SDEProblem(F!, G!, x₀, (0., 80.), p)

sol = solve(prob)