include("terminal.jl")
include("backward.jl")

const ΔΛ = [0., 0.06, 0.08];
const Ω = 2 .* 10 .^(-4:1/2:-1);

const preferences = EpsteinZin();
const jump = Jump()
const calibration = load_object(joinpath(DATAPATH, "calibration.jld2"));
const N = 21;

# Grid construction
for ωᵣ ∈ Ω
    # Construct model
    economy = Economy(ωᵣ = ωᵣ)
    defhogg = Hogg()

    # Construct Grid
    ydomain = log.(economy.Y₀ .* (0.5, 2.))
    Tdomain = defhogg.T₀ .+ (0., 9.)
    mdomain = log.(defhogg.M₀ .* (1., 2.))
    G = RegularGrid([Tdomain, mdomain, ydomain], N)

    # Jump process
    @printf("Jump simulation with ωᵣ = %.5f\n", ωᵣ)
    jumpmodel = ModelBenchmark(preferences, economy, defhogg, jump, calibration)
    computeterminal(jumpmodel, G; verbose = true, withsave = true, datapath = DATAPATH, alternate = false, tol = 1e-2, maxiter = 2_000)
    computevalue(jumpmodel, G; cache = true, verbose = true)

    for Δλ ∈ ΔΛ
        @printf("Albedo simulation with Δλ = %.2f, ωᵣ = %.5f\n", Δλ, ωᵣ)
        # Construct model
        albedo = Albedo(λ₂ = Albedo().λ₁ - Δλ)
        hogg = calibrateHogg(albedo)
        model = ModelInstance(preferences, economy, hogg, albedo, calibration)
        
        computeterminal(model, G; verbose = true, withsave = true, datapath = DATAPATH, alternate = false, tol = 1e-2, maxiter = 2_000)
        computevalue(model, G; verbose = true, cache = true)
    end
end