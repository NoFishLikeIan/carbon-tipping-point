using DotEnv, JLD2
using Model, Grid

using Interpolations: Extrapolation

include("../utils/saving.jl")
include("../utils/simulating.jl")

begin # Global variables
    env = DotEnv.config(".env")

    DATAPATH = get(env, "DATAPATH", "data")
    
    datapath = joinpath(DATAPATH, get(env, "SIMULATIONPATH", "simulaton"))
    negdatapath = joinpath(datapath, "negative")
    experimentpath = joinpath(DATAPATH, "experiments")

    PLOTPATH = get(env, "PLOTPATH", "plots")
    PRESENTATIONPATH = joinpath(PLOTPATH, "presentation")

    SEED = 11148705
end;

begin # Construct models and grids
    thresholds = [1.5, 2.5];
    Ψ = [0.75, 1.5]
    Θ = [10.]
    Ρ = [0., 1e-3]
    Ωᵣ = [0., 0.017558043747351086]

	calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
    damages = GrowthDamages()
    hogg = Hogg()

    tippingmodels = TippingModel[]
    jumpmodels = JumpModel[]

    for ψ ∈ Ψ, θ ∈ Θ, ϱ ∈ Ρ, ωᵣ ∈ Ωᵣ    
        preferences = EpsteinZin(θ = θ, ψ = ψ);
        economy = Economy(ϱ = ϱ, ωᵣ = ωᵣ)

        for Tᶜ ∈ thresholds
            albedo = Albedo(Tᶜ)
            model = TippingModel(albedo, hogg, preferences, damages, economy, calibration)
        
            push!(tippingmodels, model)
        end

        jump = Jump()
        jumpmodel = JumpModel(jump, hogg, preferences, damages, economy, calibration)

        push!(jumpmodels, jumpmodel)
    end
    
    models = AbstractModel[tippingmodels...] # jumpmodels...]
end;

begin # Interpolated policies and values
    itpmap = Dict{Symbol, Dict{AbstractModel, Dict{Symbol, Extrapolation}}}()

    paths = [(datapath, :constrained)] # FIXME: Do the negative interpolation; [(datapath, :constrained), (negdatapath, :negative)]

    for (path, symb) in paths
        results = loadtotal.(models; datapath = path)
        interpolations = buildinterpolations.(results)

        itpmap[symb] = Dict{AbstractModel, Dict{Symbol, Extrapolation}}(models .=> interpolations)
    end
end;

