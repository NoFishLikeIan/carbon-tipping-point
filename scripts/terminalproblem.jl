using Revise
using JLD2

using Model

function vfunc(x::Model.Point)
    economy = Economy()
    hogg = Hogg()

    -100f0 + economy.Y₀ * ((x.y / log(economy.Y₀))^2 - Model.d(x.T, economy, hogg))
end

function terminaliteration(N::Int; tol = 1f-3, maxiter = 100, verbose = false)
    @load "data/calibration.jld2" calibration
    domain = [
        (Hogg().Tᵖ, Hogg().T̄), 
        (log(Hogg().M₀), log(Hogg().M̄)), 
        (log(Economy().Y̲), log(Economy().Ȳ))
    ]

    model = ModelInstance(calibration = calibration, grid = RegularGrid(domain, N));

    V₀ = vfunc.(model.grid.X);
    policy = similar(V₀)

    verbose && println("Starting iterations...")

    Vᵢ, Vᵢ₊₁ = copy(V₀), copy(V₀);
    for iter in 1:maxiter
        terminaljacobi!(Vᵢ₊₁, policy, model)
        ε = maximum(abs.(Vᵢ₊₁ - Vᵢ))
        if ε < tol
            return Vᵢ₊₁, policy
        end
        
        Vᵢ .= Vᵢ₊₁
        verbose && print("Iteration $iter / $maxiter, ε = $ε...\r")
    end

    verbose && println("Done without convergence.")
    return Vᵢ₊₁, policy
end

V̄, policy = terminaliteration(101; verbose = true, maxiter = 1000);