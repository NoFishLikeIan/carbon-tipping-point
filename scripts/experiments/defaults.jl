using DotEnv, JLD2
using Model, Grid

using Interpolations: Extrapolation

include("../utils/saving.jl")
include("../utils/simulating.jl")

begin # Global variables
    env = DotEnv.config(".env")
    BASELINE_YEAR = 2020

    DATAPATH = get(env, "DATAPATH", "data")
    
    datapath = joinpath(DATAPATH, get(env, "SIMULATIONPATH", "simulaton"))
    negdatapath = joinpath(datapath, "negative")
    experimentpath = joinpath(DATAPATH, "experiments")

    PLOTPATH = get(env, "PLOTPATH", "plots")
    PRESENTATIONPATH = joinpath(PLOTPATH, "presentation")

    SAVEFIG = false
    LINE_WIDTH = 2.5
    SEED = 11148705
end;

begin # Construct models and grids
    thresholds = [1.5, 2.5];

	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
	preferences = EpsteinZin()
    damages = GrowthDamages()
    economy = Economy()
    hogg = Hogg()
    
	tippingmodels = TippingModel[]

	for Tᶜ ∈ thresholds
	    albedo = Albedo(Tᶜ)
	    model = TippingModel(albedo, hogg, preferences, damages, economy, calibration)

		push!(tippingmodels, model)
	end

    jumpmodel = JumpModel(Jump(), hogg, preferences, damages, economy, calibration)

    models = AbstractModel[tippingmodels..., jumpmodel]
end;

begin # Interpolated policies and values
    itpmap = Dict{Symbol, Dict{AbstractModel, Dict{Symbol, Extrapolation}}}()

    for (path, symb) in [(datapath, :constrained), (negdatapath, :negative)]
        results = loadtotal.(models; datapath = path)
        interpolations = buildinterpolations.(results)

        itpmap[symb] = Dict{AbstractModel, Dict{Symbol, Extrapolation}}(models .=> interpolations)
    end
end;

