function constructb!(b, valuefunction::ValueFunction, Δt⁻¹, model::M, G::RegularGrid{N₁,N₂,S}) where {N₁,N₂,S,D<:Damages{S},P<:LogSeparable{S},M<:AbstractModel{S,D,P}}
    @unpack economy, preferences = model
    χ = χopt(valuefunction.t.t, economy, preferences)
    dy = preferences.ρ * log(χ) + economy.ϱ + ϕ(valuefunction.t.t, χ, economy) - preferences.θ * economy.σₖ^2 / 2
    ωₜ = ω(valuefunction.t.t, economy)

    @inbounds for k in eachindex(valuefunction.H)
        Xᵢ = G.X[k]
        εᵢ = valuefunction.ε[k]
        
        u = (1 - preferences.θ) * (dy - d(Xᵢ.T, Xᵢ.m, model) - ωₜ * εᵢ^2 / 2)

        b[k] = u + Δt⁻¹ * valuefunction.H[k]
    end
end

function updateε!(valuefunction, ∂ₘH, model, G)
    @inbounds for k in eachindex(valuefunction.ε)
        valuefunction.ε[k] = εopt(valuefunction.t.t, G.X[k], ∂ₘH[k], model)
    end
end

function initialiseproblem(valuefunction::ValueFunction, Δt⁻¹, model::M, G::RegularGrid{N₁,N₂,S}, calibration::Calibration) where {N₁,N₂,S,D<:Damages{S},P<:LogSeparable{S},M<:AbstractModel{S,D,P}}
    A₀ = (model.preferences.ρ + Δt⁻¹) * I - constructA(V, model, G, calibration)
    b₀ = Vector{S}(undef, length(G)); constructb!(b₀, valuefunction, Δt⁻¹, model, G)

    return LinearSolve.init(LinearProblem(A₀, b₀), KLUFactorization())
end

"Iterate linear solver until convergence"
function steadystate!(valuefunction::ValueFunction{S, N₁, N₂}, Δt::S, model, G, calibration; iterations = 10_000, printevery = iterations ÷ 1, tolerance::Error{S} = Error{S}(1e-5, 1e-5), verbose = 0) where {S, N₁, N₂}
    Δt⁻¹ = 1 / Δt

    for iter in 1:iterations
        problem = initialiseproblem(valuefunction, Δt⁻¹, model, G, calibration)
        solve!(problem)
        
        itererror = abserror(problem.u, valuefunction.H)
        ∂ₘH .= ∂ᵐ * problem.u
        updateε!(valuefunction, ∂ₘH, model, G)

        valuefunction.H .= reshape(problem.u, size(G))

        if itererror < tolerance
            return valuefunction, (iter, itererror)
        end

        if verbose > 0 && iter % printevery == 0
            @printf "Iteration %d / %d: absolute error = %.6e, relative error = %.6e\r" iter iterations itererror.absolute itererror.relative
        end

        if verbose > 1
            @printf "Iteration %d / %d: absolute error = %.6e, relative error = %.6e\r" iter iterations itererror.absolute itererror.relative
        end
    end

    @warn "Failed convergence in $iterations iterations."
    return valuefunction, (iterations, Error{S}(Inf, Inf))
end