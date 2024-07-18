include("terminal.jl")
include("backward.jl")

ΔΛ = [0.06, 0.08];
N = 51;
VERBOSE = true

# Construct model
preferences = EpsteinZin();
jump = Jump()
calibration = load_object(joinpath(DATAPATH, "calibration.jld2"));

economy = Economy()
hogg = Hogg()
damages = GrowthDamages()

# Construct Grid
Tdomain = hogg.T₀ .+ (0., 7.)
mdomain = (
    mstable(Tdomain[1] - 0.75, hogg, albedo), mstable(Tdomain[2], hogg, albedo)
)
domains = [Tdomain, mdomain]

G = RegularGrid(domains, N);

# Jump process
@printf("Jump simulation")
jumpmodel = ModelBenchmark(preferences, economy, damages, hogg, jump, calibration)
computeterminal(jumpmodel, G; verbose = VERBOSE, withsave = true, datapath = DATAPATH, tol = 1e-4, maxiter = 20_000, alternate = true)
computevalue(jumpmodel, G; cache = true, verbose = VERBOSE)

for Δλ ∈ ΔΛ
    @printf("Albedo simulation with Δλ = %.2f\n", Δλ)
    # Construct model
    albedo = Albedo(λ₂ = Albedo().λ₁ - Δλ)
    hogg = calibrateHogg(albedo)
    model = ModelInstance(preferences, economy, damages, hogg, albedo, calibration)
    
    computeterminal(model, G; verbose = VERBOSE, withsave = true, datapath = DATAPATH, alternate = true, tol = 1e-4, maxiter = 20_000)
    computevalue(model, G; verbose = VERBOSE, cache = true)
end
