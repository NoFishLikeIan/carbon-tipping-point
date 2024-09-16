using DotEnv: config
include("utils/saving.jl")

# Distributed processing
using Distributed: nprocs, addprocs
ADDPROCS = getnumber(env, "ADDPROCS", 0; type = Int)
addprocs(ADDPROCS; exeflags="--project") # A bit sad that I have to do this

VERBOSE && "Running with $(nprocs()) processor..."

include("markov/terminal.jl")
include("markov/backward.jl")

# Construct model
calibration = load_object(joinpath(DATAPATH, "calibration.jld2"));
hogg = Hogg()
damages = GrowthDamages()
jump = Jump()

# Construct Grid
Tdomain = hogg.Tᵖ .+ (0., 9.);
mdomain = mstable.(Tdomain, hogg)
G = RegularGrid([Tdomain, mdomain], N)

for ψ ∈ Ψ, θ ∈ Θ, ϱ ∈ Ρ, ωᵣ ∈ Ωᵣ
    preferences = EpsteinZin(θ = θ, ψ = ψ);
    economy = Economy(ϱ = ϱ, ωᵣ = ωᵣ)
    jumpmodel = JumpModel(jump, hogg, preferences, damages, economy, calibration)


    VERBOSE && println("\nSolving jump model $(ALLOWNEGATIVE ? "with" : "without") negative emission...")
    if RUNTERMINAL
        Gterminal = terminalgrid(N, jumpmodel)
        computeterminal(jumpmodel, Gterminal; verbose = VERBOSE, datapath = datapath, alternate = true, tol = TOL, overwrite = OVERWRITE)
    end

    if RUNBACKWARDS
        VERBOSE && println("Running backward simulation...")
        computebackward(jumpmodel, G; verbose = VERBOSE, datapath = datapath, overwrite = OVERWRITE, tstop = TSTOP, cachestep = CACHESTEP)
    end
end