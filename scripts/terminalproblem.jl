using Revise
using JLD2

using Model

function terminaliteration(N::Int; kwargs...)
    @load "data/calibration.jld2" calibration
    hogg = Hogg()
    economy = Economy(ρ = 0.02)
    albedo = Albedo()

    domain = [
        (hogg.Tᵖ, hogg.Tᵖ + 10.), 
        (log(hogg.M₀), Model.mstable(hogg.Tᵖ + 10., hogg, albedo)), 
        (log(economy.Y₀ / 2), log(economy.Y₀ * 2))
    ]

    model = ModelInstance(
        calibration = calibration, 
        grid = RegularGrid(domain, N),
        hogg = hogg, albedo = albedo, economy = economy
    );

    V₀ = -ones(size(model.grid))

    terminaliteration(V₀, model; kwargs...)
end
function terminaliteration(V₀::Array{Float64, 3}, model::ModelInstance; tol = 1e-3, maxiter = 100_000, verbose = false)
    policy = similar(V₀)

    verbose && println("Starting iterations...")

    Vᵢ, Vᵢ₊₁ = copy(V₀), copy(V₀);
    for iter in 1:maxiter

        terminaljacobi!(Vᵢ₊₁, policy, model; fwdpass = true)
        ε = maximum(abs.(Vᵢ₊₁ - Vᵢ))
        if ε < tol
            return Vᵢ₊₁, policy, model
        end
        
        Vᵢ .= Vᵢ₊₁
        verbose && print("Iteration $iter / $maxiter, ε = $ε...\r")
    end

    verbose && println("\nDone without convergence.")
    return Vᵢ₊₁, policy, model
end

V̄, policy, model = terminaliteration(30; verbose = true, maxiter = 10_000, tol = 1e-5);

println("\nSaving solution into data/terminal.jld2...")
@save "data/terminal.jld2" V̄ policy model