"Backward simulates from V̄ = V(τ) down to V(0). Stores nothing."
function backwardsimulation!(V::AbstractArray{Float64, 3}, policy::AbstractArray{Policy, 3}, model::ModelInstance; verbose = false, cachepath = nothing, tmin = 0.)
    @unpack grid, economy, hogg, albedo, calibration = model
    σₜ² = (hogg.σₜ / (hogg.ϵ * grid.Δ.T))^2
    σₖ² = (economy.σₖ / grid.Δ.y)^2

    indices = CartesianIndices(grid)
    L, R = extrema(indices)

    timequeue = PriorityQueue{typeof(L), Float64}(
        Base.Order.Reverse, indices .=> economy.τ);

    verbose && println("Starting backward simulation...")
    cache = !isnothing(cachepath)
    
    if cache
        tcache = economy.τ - 1.
        cachefile = jldopen(cachepath, "a+")
    end

    while !isempty(timequeue)
        idx, tᵢ = dequeue_pair!(timequeue)
        Xᵢ = grid.X[idx]
        d = driftbounds(tᵢ, Xᵢ, model)
        verbose && print("Remaining values $(length(timequeue)), time $tᵢ\r")

        Qᵢ = σₜ² + σₖ² + grid.h * sum(abs(d))
        Δtᵢ = (grid.h)^2 / Qᵢ 

        # Neighbouring nodes
        VᵢT₊, VᵢT₋ = V[min(idx + I[1], R)], V[max(idx - I[1], L)]
        Vᵢm₊ = V[min(idx + I[2], R)]
        Vᵢy₊, Vᵢy₋ = V[min(idx + I[3], R)], V[max(idx - I[3], L)]

        # Optimal control
        optpolicy = optimalpolicy(tᵢ, Xᵢ, V[idx], Vᵢy₊, Vᵢm₊, Vᵢy₋, model)

        # -- Temperature
        PT₊ = ((σₜ² / 2.) + grid.h * max(d.dT, 0.)) / Qᵢ
        PT₋ = ((σₜ² / 2.) + grid.h * max(-d.dT, 0.)) / Qᵢ

        # -- Carbon concentration
        dm = d.dm - (optpolicy.α / grid.Δ.m)
        Pm₊ = grid.h * dm / Qᵢ

        # -- GDP
        dy = b(tᵢ, Xᵢ, optpolicy, model) / grid.Δ.y
        Py₊ = ((σₖ² / 2.) + grid.h * max(dy, 0.)) / Qᵢ
        Py₋ = ((σₖ² / 2.) + grid.h * max(-dy, 0.)) / Qᵢ

        # -- Residual
        P = Py₊ + Py₋ + PT₊ + PT₋ + Pm₊

        V[idx] = (
            PT₊ * VᵢT₊ + PT₋ * VᵢT₋ +
            Py₊ * Vᵢy₊ + Py₋ * Vᵢy₋ +
            Pm₊ * Vᵢm₊ ) / P +
            Δtᵢ * f(optpolicy.χ, Xᵢ.y, V[idx], economy)

        policy[idx] = optpolicy

        if tᵢ > tmin
            push!(timequeue, idx => tᵢ - Δtᵢ)
        end 
        
        if cache && tᵢ ≤ tcache 
            verbose && println("\nSaving cache at $(floor(Int, tcache))...")
            g = Group(cachefile, "$(floor(Int, tcache))")
            g["V"] = V
            g["policy"] = policy
            tcache -= 1.      
        end
    end

    if cache
        close(cachefile)
    end

    return V, optpolicy
end