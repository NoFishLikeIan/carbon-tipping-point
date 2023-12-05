function optimalpolicy(t, Xᵢ::Point, Vᵢ, Vᵢy₊, Vᵢy₋, Vᵢm₊, model::ModelInstance; policy₀ = [0.01, 0.01])
    @unpack economy, hogg, albedo, calibration, grid = model
    γₜ = γ(t, economy, calibration)
    dc = TwiceDifferentiableConstraints([0., 0.], [1., γₜ])

    function objective!(z, ∇, H, u)
        χ, α = u
        bᵢ = b(t, Xᵢ, Policy(χ, α), model)
        bsgn = sign(bᵢ)
        Vᵢy = ifelse(bᵢ > 0, Vᵢy₊, Vᵢy₋)

        M = exp(Xᵢ.m)
        Aₜ = A(t, economy)
        εₜ = ε(t, M, α, model)
        εₜ′ = ε′(t, M, model)

        fᵢ, Y∂fᵢ, Y²∂²fᵢ = epsteinzinsystem(χ, Xᵢ.y, Vᵢ, economy) .* grid.h

        if !isnothing(∇) 
            ∇[1] = -Y∂fᵢ - bsgn * ϕ′(t, χ, economy) * Vᵢy
            ∇[2] = Vᵢm₊ + bsgn * Aₜ * β′(t, εₜ, model.economy) * εₜ′ * Vᵢy
        end

        if !isnothing(H)
            H[1, 2] = 0.
            H[2, 1] = 0.
            
            H[1, 1] = -Y²∂²fᵢ + bsgn * economy.κ * Aₜ^2 * Vᵢy
            H[2, 2] = bsgn * Aₜ * (εₜ′)^2 * exp(-economy.ωᵣ * t) * Vᵢy
        end
        
        if !isnothing(z)
            z = α * Vᵢm₊ - abs(bᵢ) * Vᵢy - fᵢ
            return z
        end
    end

    df = TwiceDifferentiable(only_fgh!(objective!), policy₀)
    
    res = optimize(df, dc, policy₀, IPNewton())
    
    return Policy(minimizer(res)), -minimum(res)
end

function optimalterminalpolicy(Xᵢ::Point, Vᵢ, Vᵢy₊, Vᵢy₋, model::ModelInstance; tol = 1e-3)
    @unpack economy, hogg, albedo, calibration, grid = model

    function objective(χ)
        bᵢ = bterminal(Xᵢ, χ, model) / grid.Δ.y        
        f(χ, Xᵢ.y, Vᵢ, economy) * grid.h + 
            Vᵢy₊ * max(bᵢ, 0.) + Vᵢy₋ * max(-bᵢ, 0.)
    end

    gss(objective, 0., 1.; tol = tol)
end
