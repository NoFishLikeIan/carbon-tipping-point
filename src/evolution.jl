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