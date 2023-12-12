using Model
using JLD2, DotEnv

const env = DotEnv.config()
const DATAPATH = get(env, "DATAPATH", "data/") 

function terminaliteration(V₀::Array{Float64, 3}, model::ModelInstance; tol = 1e-3, maxiter = 100_000, verbose = false)
    policy = similar(V₀)

    verbose && println("Starting iterations...")

    Vᵢ, Vᵢ₊₁ = copy(V₀), copy(V₀);
    for iter in 1:maxiter

        terminaljacobi!(Vᵢ₊₁, policy, model)
        ε = maximum(abs.((Vᵢ₊₁ - Vᵢ) ./ Vᵢ))
        if ε < tol
            return Vᵢ₊₁, policy, model
        end
        
        Vᵢ .= Vᵢ₊₁
        verbose && print("Iteration $iter / $maxiter, ε = $ε...\r")
    end

    verbose && println("\nDone without convergence.")
    return Vᵢ₊₁, policy, model
end

function computeterminal(N::Int, Δλ = 0.08; iterkwargs...)
    calibration = load_object(joinpath(DATAPATH, "calibration.jld2"))
    hogg = Hogg()
    economy = Economy()
    albedo = Albedo(λ₂ = Albedo().λ₁ - Δλ)

    domains = [
        (hogg.Tᵖ, hogg.Tᵖ + 10.), 
        (log(hogg.M₀), Model.mstable(hogg.Tᵖ + 10., hogg, albedo)), 
        (log(economy.Y₀ / 2), log(economy.Y₀ * 2))
    ]

    model = ModelInstance(
        economy = economy, hogg = hogg, albedo = albedo,
        grid = RegularGrid(domains, N),
        calibration = calibration
    )

    V₀ = -model.grid.h * ones(size(model.grid))
    V̄, policy, model = terminaliteration(V₀, model; iterkwargs...)

    savepath = joinpath(DATAPATH, "terminal", "N=$(N)_Δλ=$(Δλ).jld2")
    println("\nSaving solution into $savepath...")

    jldsave(savepath; V̄, policy, model)
end