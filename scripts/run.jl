include("terminal.jl")
include("backward.jl")

ΔΛ = [0.06, 0.08];
N = 31;

preferences = EpsteinZin();
jump = Jump()
calibration = load_object(joinpath(DATAPATH, "calibration.jld2"));

# Construct model
economy = Economy()
hogg = Hogg()
damages = GrowthDamages()

# Construct Grid
ydomain = log.(economy.Y₀ .* (0.5, 2.))
Tdomain = hogg.T₀ .+ (0., 9.)
mdomain = log.(hogg.M₀ .* (1., 2.))
G = RegularGrid([Tdomain, mdomain, ydomain], N)

# Jump process
@printf("Jump simulation")
jumpmodel = ModelBenchmark(preferences, economy, damages, hogg, jump, calibration)
computeterminal(jumpmodel, G; verbose = true, withsave = true, datapath = DATAPATH, alternate = false, tol = 1e-2, maxiter = 5_000)
computevalue(jumpmodel, G; cache = true, verbose = true)

for Δλ ∈ ΔΛ
    @printf("Albedo simulation with Δλ = %.2f\n", Δλ)
    # Construct model
    albedo = Albedo(λ₂ = Albedo().λ₁ - Δλ)
    hogg = calibrateHogg(albedo)
    model = ModelInstance(preferences, economy, damages, hogg, albedo, calibration)
    
    computeterminal(model, G; verbose = true, withsave = true, datapath = DATAPATH, alternate = false, tol = 1e-2, maxiter = 5_000)
    computevalue(model, G; verbose = true, cache = true)
end
