include("terminal.jl")
include("backward.jl")

const ΔΛ = [0., 0.06, 0.08];
const Ω = 2 .* 10 .^(-4:1/3:-1);

const preferences = EpsteinZin(θ = 3., ψ = 0.5);
const hogg = Hogg();
const calibration = load_object(joinpath(DATAPATH, "calibration.jld2"));

# Grid construction
const N = 21;
const Tdomain = hogg.T₀ .+ (0., 9.)
const mdomain = log.(hogg.M₀ .* (1., 2.))
const ydomain = log.(Economy().Y₀ .* (0.5, 2.))
const G = RegularGrid([Tdomain, mdomain, ydomain], N)

for Δλ ∈ ΔΛ, ωᵣ ∈ Ω
    @printf("Calibration Δλ = %.2f, ωᵣ = %.2f\n", Δλ, ωᵣ)
    economy = Economy(ωᵣ = ωᵣ)
    albedo = Albedo(λ₂ = 0.31 - Δλ)
    model = ModelInstance(preferences, economy, hogg, albedo, calibration)
    
    computeterminal(model, G; verbose = true, withsave = true, datapath = DATAPATH, alternate = false, tol = 1e-3, maxiter = 4_000)
    computevalue(model, G; verbose = true, cache = true)
end