include("terminal.jl")
include("backward.jl")

ΔΛ = [0., 0.06, 0.08];
N = 51;
VERBOSE = true

# Construct model
preferences = EpsteinZin();
calibration = load_object(joinpath(DATAPATH, "calibration.jld2"));
economy = Economy()
hogg = Hogg()
damages = [LevelDamages(), GrowthDamages()]

# Terminal simulation
for d in damages
    VERBOSE && println("Solving for damages = $d...")
    for Δλ ∈ ΔΛ
        VERBOSE && println("Solving albedo model Δλ = $Δλ")

        λ₂ = Albedo().λ₁ - Δλ
        albedo = Albedo(λ₂ = λ₂)
        model = TippingModel(albedo, preferences, d, economy, hogg, calibration)

        G = constructdefaultgrid(N, model)

        computeterminal(model, G; verbose = VERBOSE, datapath = DATAPATH, alternate = true, tol = 1e-3)
    end

    jumpmodel = JumpModel(Jump(), preferences, d, economy, hogg, calibration)

    G = constructdefaultgrid(N, jumpmodel)

    computeterminal(model, G; verbose = VERBOSE, datapath = DATAPATH, alternate = true, tol = 1e-3)
end
