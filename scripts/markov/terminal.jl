function constructb!(b, valuefunction::ValueFunction, Δt⁻¹, model::M, G::RegularGrid{N₁,N₂,S}, calibration) where {N₁,N₂,S,D<:Damages{S},P<:LogSeparable{S},M<:AbstractModel{S,D,P}}
    @inbounds for k in eachindex(valuefunction.H)
        Xᵢ = G.X[k]
        αᵢ = valuefunction.α[k]

        b[k] = l(valuefunction.t.t, Xᵢ, αᵢ, model, calibration) + Δt⁻¹ * valuefunction.H[k]
    end
end

function updateproblem!(problem, valuefunction::ValueFunction, Δt⁻¹, model::M, G::RegularGrid{N₁,N₂,S}, calibration::Calibration) where {N₁,N₂,S,D<:Damages{S},P<:LogSeparable{S},M<:AbstractModel{S,D,P}}
    problem.A .= constructA(valuefunction, Δt⁻¹, model, G, calibration)
    constructb!(problem.b, valuefunction, Δt⁻¹, model, G, calibration)

    return problem
end

"Iterate linear solver until convergence"
function steadystate!(valuefunction::ValueFunction{S, N₁, N₂}, Δt::S, model, G, calibration; iterations = 10_000, printevery = iterations ÷ 1, tolerance::Error{S} = Error{S}(1e-3, 1e-3), verbose = 0) where {S, N₁, N₂}
    Δt⁻¹ = 1 / Δt
    n = length(G)

    A₀ = constructA(valuefunction, Δt⁻¹, model, G, calibration)
    b₀ = Vector{S}(undef, n)
    problem = LinearSolve.init(LinearProblem(A₀, b₀), KLUFactorization())

    for iter in 1:iterations

        updateproblem!(problem, valuefunction, Δt⁻¹, model, G, calibration)
        solve!(problem)
        
        itererror = abserror(problem.u, valuefunction.H)

        valuefunction.H .= reshape(problem.u, size(G))

        if itererror < tolerance
            return valuefunction, (iter, itererror)
        end

        if (verbose > 1) || (verbose > 0 && iter % printevery == 0)
            @printf "Iteration %d: absolute error = %.2e, relative error = %.2e\r" iter itererror.absolute itererror.relative
        end

    end

    @warn "Failed convergence in $iterations iterations."
    return valuefunction, (iterations, Error{S}(Inf, Inf))
end