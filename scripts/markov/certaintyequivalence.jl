function discoveryvalues(discovery, (truevalues, truemodel, trueG), (linearvalues, linearmodel, linearG))
    effectivevalues = copy(truevalues)
    i = searchsortedfirst(linearG.ranges[1], truemodel.climate.feedback.Tᶜ + discovery)

    if i > 1
        for (t, value) in effectivevalues
            itplinearvalue = interpolateovergrid(linearvalues[t], linearG, trueG)
            value.α[1:(i - 1), :] .= itplinearvalue.α[1:(i - 1), :]
        end
    end

    return effectivevalues, truemodel, linearG
end

function constructstaticDᵐ!(stencil, αitp, t, G::RegularGrid{N₁, N₂, S}, calibration) where {N₁, N₂, S}
    γₜ = γ(t, calibration)
    
    Δm = step(G, 2)
    Tspace, mspace = G.ranges
    rows, columns, data = stencil
    counter = 1
    @inbounds for j in axes(G, 2), i in axes(G, 1)
        k = LinearIndex((i, j), G)
        x = Point(Tspace[i], mspace[j])
        
        α = αitp(x.T, x.m, t)
        bᵐ = (γₜ - α) / Δm

        z = max(bᵐ, zero(S))
        w = max(-bᵐ, zero(S))
        y = zero(S)

        if j < N₂
            rows[counter] = k; columns[counter] = LinearIndex((i, j + 1), G)
            data[counter] = z; counter += 1
            y -= z
        end

        if j > 1
            rows[counter] = k; columns[counter] = LinearIndex((i, j - 1), G)
            data[counter] = w; counter += 1
            y -= w
        end

        rows[counter] = k; columns[counter] = k
        data[counter] = y; counter += 1
    end
end

function constructstaticsource(αitp, t, H, Δt⁻¹, model, G::RegularGrid{N₁, N₂, S}, calibration) where {N₁, N₂, S}
    source = Vector{S}(undef, N₁ * N₂)
    constructstaticsource!(source, αitp, t, H, Δt⁻¹, model, G, calibration)
end

function constructstaticsource!(source, αitp, t, H, Δt⁻¹, model, G::RegularGrid{N₁, N₂, S}, calibration) where {N₁, N₂, S}
    Tspace, mspace = G.ranges
    @inbounds for j in axes(G, 2), i in axes(G, 1)
        x = Point(Tspace[i], mspace[j])
        α = αitp(x.T, x.m, t)
        Hᵢ = H[i, j]

        ΔT = step(G, 1)
        bᵀ = μ(x.T, x.m, model.climate)
        useforward = (bᵀ ≥ 0 && i < N₁) || (bᵀ < 0 && i == 1)

        ∂ᵀH = ((useforward ? H[i + 1, j] : H[i - 1, j]) - Hᵢ) / ΔT
        advection = ∂ᵀH^2 * variance(x.T, model.climate.hogg) / 2

        k = LinearIndex((i, j), G)
        source[k] = advection + l(t, x, α, model, calibration) + Δt⁻¹ * Hᵢ
    end

    return source
end

function staticbackwardstep!(problem, R, stencilm, αitp, valuefunction, Δt⁻¹, model, G, calibration)
    @unpack t, H = valuefunction
    n = length(G)

    constructstaticDᵐ!(stencilm, αitp, t.t, G, calibration)
    constructstaticsource!(problem.b, αitp, t.t, H, Δt⁻¹, model, G, calibration)
    
    A = R - sparse(stencilm[1], stencilm[2], stencilm[3], n, n)
    problem.A = A
    
    sol = solve!(problem)
    H .= reshape(sol.u, size(G))
    
    return valuefunction
end

function staticbackward!(valuefunction::ValueFunction{S, N₁, N₂}, Δt::S, αitp, model, G::RegularGrid{N₁, N₂, S}, calibration; t₀ = zero(S), verbose = 0, printstep = 10, alg = KLUFactorization()) where {S, N₁, N₂}
    @unpack preferences = model
    
    if verbose > 0
        tverbose = copy(valuefunction.t.t)
    end

    Δt⁻¹ = 1 / Δt
    n = length(G)
    stencilT, stencilm = makestencil(G)
    constructDᵀ!(stencilT, model, G)
    constructstaticDᵐ!(stencilm, αitp, valuefunction.t.t, G, calibration)
    b₀ = constructstaticsource(αitp, valuefunction.t.t, valuefunction.H, Δt⁻¹, model, G, calibration)
    Sᵨ = (preferences.ρ + Δt⁻¹) * I
    R = Sᵨ - sparse(stencilT[1], stencilT[2], stencilT[3], n, n)
    A₀ = R - sparse(stencilm[1], stencilm[2], stencilm[3], n, n)
    problem = LinearSolve.init(LinearProblem(A₀, b₀), alg)
    
    staticbackwardstep!(problem, R, stencilm, αitp, valuefunction, Δt⁻¹, model, G, calibration)
 
    while t₀ < valuefunction.t.t
        staticbackwardstep!(problem, R, stencilm, αitp, valuefunction, Δt⁻¹, model, G, calibration)

        valuefunction.t.t -= Δt

        if (verbose > 1) || (verbose > 0 && valuefunction.t.t < tverbose)
            if verbose > 0 
                tverbose = tverbose - printstep 
            end
            @printf "Time %.2f\r" valuefunction.t.t
        end
    end

    return valuefunction
end