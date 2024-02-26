include("terminal.jl")
include("backward.jl")

const ΔΛ = [0., 0.06, 0.08];
const Ω = 2 .* 10 .^(-4:1/3:-1);

const preferences = EpsteinZin();
const calibration = load_object(joinpath(DATAPATH, "calibration.jld2"));
const N = 21;

# Grid construction
for Δλ ∈ ΔΛ, ωᵣ ∈ Ω
    @printf("Calibration Δλ = %.2f, ωᵣ = %.2f\n", Δλ, ωᵣ)

# Construct model
    economy = Economy(ωᵣ = ωᵣ)
    albedo = Albedo(λ₂ = Albedo().λ₁ - Δλ)
    hogg = calibrateHogg(albedo)
    model = ModelInstance(preferences, economy, hogg, albedo, calibration)
        
    # Construct Grid
    Tdomain = hogg.T₀ .+ (0., 9.)
    mdomain = log.(hogg.M₀ .* (1., 2.))
    ydomain = log.(economy.Y₀ .* (0.5, 2.))
    G = RegularGrid([Tdomain, mdomain, ydomain], N)
    
    computeterminal(model, G; verbose = true, withsave = true, datapath = DATAPATH, alternate = false, tol = 1e-2, maxiter = 2_000)
    computevalue(model, G; verbose = true, cache = true)
end