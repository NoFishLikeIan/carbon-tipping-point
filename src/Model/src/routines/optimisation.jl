function optimalpolicy(t, Xᵢ::Point, Vᵢ, Vᵢy₊, Vᵢy₋, Vᵢm₊, model::ModelInstance; options::Options = Options(allow_f_increases = true, successive_f_tol = 2), p₀ = Policy(1e-5, 1e-5))
    γₜ = γ(t, model.economy, model.calibration)
    dc = TwiceDifferentiableConstraints([0., 0.], [1., γₜ])
    
    function objective!(z, ∇, H, u)
        bᵢ = b(t, Xᵢ, u[1], u[2], model) / model.grid.Δ.y
        bsgn = sign(bᵢ)
        Vᵢy = ifelse(bᵢ > 0, Vᵢy₊, Vᵢy₋)

        M = exp(Xᵢ.m)
        Aₜ = A(t, model.economy)
        εₜ = ε(t, M, u[2], model)
        εₜ′ = ε′(t, M, model)

        fsystem = epsteinzinsystem(u[1], Xᵢ.y, Vᵢ, model.economy)

        if !isnothing(∇) 
            ∇[1] = -fsystem[2] * model.grid.h - bsgn * ϕ′(t, u[1], model.economy) * Vᵢy
            ∇[2] = (Vᵢm₊ / model.grid.Δ.m) + bsgn * Aₜ * β′(t, εₜ, model.economy) * εₜ′ * Vᵢy
        end

        if !isnothing(H)
            H[1, 2] = 0.
            H[2, 1] = 0.
            
            H[1, 1] = -fsystem[3] * model.grid.h + bsgn * model.economy.κ * (Aₜ)^2 * Vᵢy
            H[2, 2] = bsgn * Aₜ * (εₜ′)^2 * exp(-model.economy.ωᵣ * t) * Vᵢy
        end
        
        if !isnothing(z)
            return u[2] * (Vᵢm₊ / model.grid.Δ.m) - abs(bᵢ) * Vᵢy - fsystem[1] * model.grid.h
        end
    end

    df = TwiceDifferentiable(only_fgh!(objective!), [0., 0.])
    
    res = optimize(df, dc, [p₀.χ, p₀.α], IPNewton(), options)
    u = minimizer(res)
    
    return Policy(u[1], u[2])
end

function optimalterminalpolicy(Xᵢ::Point, Vᵢ, Vᵢy₊, Vᵢy₋, model::ModelInstance; tol = 1e-15)
    @unpack economy, hogg, albedo, calibration, grid = model

    function objective(χ)
        bᵢ = bterminal(Xᵢ, χ, model) / grid.Δ.y        
        f(χ, Xᵢ.y, Vᵢ, economy) * grid.h + 
            Vᵢy₊ * max(bᵢ, 0.) + Vᵢy₋ * max(-bᵢ, 0.)
    end

    gss(objective, 0., 1.; tol = tol)
end
